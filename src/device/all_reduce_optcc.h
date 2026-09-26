/*************************************************************************
 * OptCC AllReduce with one straggler (arXiv:2606.01680, Sec. 4.1-4.2), as
 * NCCL_ALGO_OPTCCRING x NCCL_PROTO_SIMPLE. Included from all_reduce.h.
 *
 * Each loop of a channel is one segment of nh = p-1 sections (chunks). Healthy
 * index i owns section i. Four stages per segment:
 *   S1 reduce-scatter over the healthy ring   S2 owner -> straggler
 *   S3 straggler adds its input, returns       S4 allgather over the healthy ring
 * Ordering 1 (even channels, patterns A/C): S1 S2 S3 S4.
 * Ordering 2 (odd channels, patterns B/D): S3 S1 S4 S2 -- the straggler sends
 * its raw section j to the rank that starts j's reduce-scatter chain, which
 * folds it in with recvReduceSend on its first hop; the owner returns the
 * global sum last. Channels run concurrently, so the healthy ring of one
 * channel overlaps the straggler link of another, as the paper's patterns do.
 *
 * Every stage is a Primitives object with one recv and one send peer, built
 * in its own scope: its destructor writes the connection steps back, so the
 * next stage continues on the same connections. Per loop and connection the
 * step counts match ncclProxySaveOp's ncclPatternOptcc case:
 *   healthy: 2(nh-1) chunks to next / from prev, 1 to / 1 from the straggler;
 *   straggler: 1 to / 1 from each healthy rank.
 ************************************************************************/

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_OPTCCRING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>;
    // FanAsymmetric: send-only / recv-only stages pass -1 for the other side,
    // and FanSymmetric would take the recv count for both.
    using Prims = Primitives<T, RedOp, FanAsymmetric<1, 1>, 1, Proto, 0>;
    struct ncclOptccRing* oc = &ncclShmem.channel.optccRing;
    const int nh = oc->nHealthyRanks;
    const int hi = oc->healthyIndex;
    int S = oc->straggler, prev = oc->ringPrev, next = oc->ringNext, none = -1;
    const int ordering = (ncclShmem.channelId % 2 == 0) ? 1 : 2;

    ssize_t gridOffset, channelCount, chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), (ssize_t*)nullptr, &gridOffset, &channelCount, &chunkCount);
    const ssize_t loopCount = nh * chunkCount;

    for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
      ssize_t remCount = channelCount - elemOffset;
      if (remCount < loopCount) chunkCount = alignUp(divUp(remCount, nh), 16/sizeof(T));
      auto mod = [&]__device__(int j)->int { return ((j % nh) + nh) % nh; };
      // Section j of this segment: offset and length (<= 0 on an empty tail;
      // the primitive still runs so every connection advances the same steps).
      auto off = [&]__device__(int j)->ssize_t { return gridOffset + elemOffset + (ssize_t)mod(j) * chunkCount; };
      auto len = [&]__device__(int j)->int { return (int)min(chunkCount, remCount - (ssize_t)mod(j) * chunkCount); };

      if (hi >= 0) {
        // ---- healthy rank ----
        if (ordering == 2) {  // S3 + first hop of S1: straggler's raw section (hi-1) + ours -> next
          Prims p(tid, nthreads, &S, &next, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);
          p.recvReduceSend(off(hi-1), len(hi-1));
        }
        {
          Prims ring(tid, nthreads, &prev, &next, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);
          // S1 reduce-scatter (runRing's steps with nranks = nh, ringIx = hi)
          if (ordering == 1) ring.send(off(hi-1), len(hi-1));
          for (int j = 2; j < nh; ++j) ring.recvReduceSend(off(hi-j), len(hi-j));
          ring.recvReduceCopy(off(hi), off(hi), len(hi));  // ordering 2: already the global sum
        }
        if (ordering == 1) {  // S2 + S3: partial up, global sum back into our section
          Prims link(tid, nthreads, &S, &S, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);
          link.sendFromOutput(off(hi), len(hi));
          link.recv(off(hi), len(hi));
        }
        {
          Prims ring(tid, nthreads, &prev, &next, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);
          // S4 allgather from the owned section
          ring.sendFromOutput(off(hi), len(hi));
          for (int j = 1; j < nh - 1; ++j) ring.recvCopySend(off(hi-j), len(hi-j));
          ring.recv(off(hi+1), len(hi+1));
        }
        if (ordering == 2) {  // S2 last: global sum to the straggler
          Prims link(tid, nthreads, &none, &S, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);
          link.sendFromOutput(off(hi), len(hi));
        }
      } else {
        // ---- straggler: one flow at a time, in healthy order ----
        for (int i = 0; i < nh; ++i) {
          int h = oc->healthy[i];
          if (ordering == 1) {  // S2: partial of section i in, + our input, kept in the output
            Prims p(tid, nthreads, &h, &none, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);
            p.recvReduceCopy(off(i), off(i), len(i));
          } else {  // S3: raw section i to the rank that starts its chain, healthy index i+1
            int starter = oc->healthy[mod(i+1)];
            Prims p(tid, nthreads, &none, &starter, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);
            p.send(off(i), len(i));
          }
        }
        if (ordering == 1) {
          // S3 as its own stage, after all of S2, as in the paper: with the arbiter
          // serving sends B, D before A, C, a fused receive-and-return stalled the
          // receives of A / C until their returns were allowed (DESIGN.md 10).
          for (int i = 0; i < nh; ++i) {
            int h = oc->healthy[i];
            Prims p(tid, nthreads, &none, &h, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);
            p.sendFromOutput(off(i), len(i));
          }
        }
        if (ordering == 2) {
          for (int i = 0; i < nh; ++i) {  // S2: global sum of section i from its owner
            int h = oc->healthy[i];
            Prims p(tid, nthreads, &h, &none, work->sendbuff, work->recvbuff, work->redOpArg, 0, 0, 0, work);
            p.recv(off(i), len(i));
          }
        }
      }
    }
  }
};
