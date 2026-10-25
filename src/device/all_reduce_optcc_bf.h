/*************************************************************************
 * OptCC AllReduce with bubble filling, for l < 2 - 1/(p-1) (paper Sec. 4.2,
 * Fig. 7, App. C). Included from all_reduce_optcc.h, which runs it when
 * NCCL_OPTCC_BF_NUM > 0.
 *
 * The schedule is OptccRingOrig's (all_reduce_optcc.h): four stages per
 * segment, ordering 1 on even channels and ordering 2 on odd ones. On top of
 * it, a point-to-point AllReduce of one block per segment through the
 * straggler, on its own connection (connIndex 1, ncclTransportOptccConnect):
 * own buffers, FIFO and QPs, and outside the straggler's SERIAL arbiter, so a
 * block never queues in front of a section.
 *   - healthy: section to the straggler (connIndex 0), then its raw block j
 *     (connIndex 1); at the end of the loop, the sum of block j-1 (connIndex 1);
 *   - straggler: all nh section exchanges first (connIndex 0, healthy order),
 *     then the sums of block j-1 back, then this loop's nh raw blocks added
 *     into its own (connIndex 1). Block traffic fills the time the straggler
 *     waits for the next loop's sections; a block's sum is needed only then.
 *
 * A loop is nh full sections (one chunk each, like OptccRingOrig) plus a
 * block of r chunks (r = NUM / DEN <= 1, ncclOptccBlkCount; the host's loop
 * count uses the same size). A short last loop shares out what is left: equal
 * sections, and a block only past nh full ones. Loop j returns the sum of block
 * j-1, and a flush after the last loop returns the last one.
 *
 * A primitive call always moves its connections by one chunk, even when empty.
 * Per loop and connection the chunk counts match ncclProxySaveOp's
 * ncclPatternOptcc case (on connIndex 1 the flush stands in for loop 0's
 * missing sum):
 *   healthy: 2(nh-1) to next / from prev; 1 to / 1 from the straggler on each
 *            of connIndex 0 and 1;
 *   straggler: 1 to / 1 from each healthy rank on each of connIndex 0 and 1.
 ************************************************************************/

template <typename T, typename RedOp>
struct OptccRingBf {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS / ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>;
    // FanAsymmetric: send-only / recv-only stages pass -1 for the other side,
    // and FanSymmetric would take the recv count for both.
    using Prims = Primitives<T, RedOp, FanAsymmetric<1, 1>, 1, Proto, 0>;
    struct ncclOptccRing* oc = &ncclShmem.channel.optccRing;
    const int nh = oc->nHealthyRanks;
    const int hi = oc->healthyIndex;
    int S = oc->straggler, prev = oc->ringPrev, next = oc->ringNext, none = -1;
    const int ordering = (ncclShmem.channelId % 2 == 0) ? 1 : 2;
    void* in = work->sendbuff;
    void* out = work->recvbuff;

    ssize_t gridOffset, channelCount, chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), (ssize_t*)nullptr, &gridOffset, &channelCount,
                    &chunkCount);
    const ssize_t sec0 = chunkCount;
    const ssize_t blkMax = ncclOptccBlkCount(chunkCount, oc->bfNum, oc->bfDen, sizeof(T));
    const ssize_t loopCount = nh * sec0 + blkMax;
    const ssize_t align = 16 / sizeof(T);
    auto mod = [&] __device__(int j) -> int { return ((j % nh) + nh) % nh; };
    // Block of the previous loop, whose sum this loop returns.
    ssize_t prevOff = gridOffset;
    int prevN = 0;
    // Straggler: the previous block's sum back to each healthy rank (connIndex 1).
    auto sendSums = [&] __device__() {
      for (int i = 0; i < nh; ++i) {  // ordering 2: to the rank that starts section i's chain
        int h = ordering == 1 ? oc->healthy[i] : oc->healthy[mod(i + 1)];
        Prims q(tid, nthreads, &none, &h, out, out, work->redOpArg, 0, 0, 1, work);
        q.sendFromOutput(prevOff, prevN);
      }
    };

    for (ssize_t elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
      const ssize_t remCount = channelCount - elemOffset;
      const bool first = elemOffset == 0;
      // Full sections and the block; a short last loop shares out what is left
      // (sections <= a chunk, since chunkCount is a multiple of align).
      ssize_t sec = sec0;
      if (remCount < (ssize_t)nh * sec0) sec = alignUp(divUp(remCount, nh), align);
      const int blkN = (int)max((ssize_t)0, min(blkMax, remCount - (ssize_t)nh * sec));
      const ssize_t blkOff = gridOffset + elemOffset + (ssize_t)nh * sec;
      // Section j of this segment: offset and length (<= 0 on an empty tail;
      // the primitive still runs so every connection advances the same steps).
      auto off = [&] __device__(int j) -> ssize_t { return gridOffset + elemOffset + (ssize_t)mod(j) * sec; };
      auto len = [&] __device__(int j) -> int { return (int)min(sec, remCount - (ssize_t)mod(j) * sec); };

      // Primitives(..., group, connIndexRecv, connIndexSend, work): sections on
      // connIndex 0, the blocks and their sums on connIndex 1 (own buffers, FIFO
      // and QPs, outside the straggler's arbiter).
      if (hi >= 0) {
        // ---- healthy rank ----
        if (ordering == 2) {  // S3 + first hop of S1: straggler's raw section (hi-1) + ours -> next
          Prims p(tid, nthreads, &S, &next, in, out, work->redOpArg, 0, 0, 0, work);
          p.recvReduceSend(off(hi - 1), len(hi - 1));
        }
        {
          Prims ring(tid, nthreads, &prev, &next, in, out, work->redOpArg, 0, 0, 0, work);
          // S1 reduce-scatter (runRing's steps with nranks = nh, ringIx = hi)
          if (ordering == 1) ring.send(off(hi - 1), len(hi - 1));
          for (int j = 2; j < nh; ++j) ring.recvReduceSend(off(hi - j), len(hi - j));
          ring.recvReduceCopy(off(hi), off(hi), len(hi));  // ordering 2: already the global sum
        }
        if (ordering == 1) {  // S2 + S3: partial up, global sum back into our section
          {
            Prims up(tid, nthreads, &none, &S, in, out, work->redOpArg, 0, 0, 0, work);
            up.sendFromOutput(off(hi), len(hi));
          }
          {  // our raw block, on its own connection
            Prims blk(tid, nthreads, &none, &S, in, out, work->redOpArg, 0, 0, 1, work);
            blk.send(blkOff, blkN);
          }
          Prims down(tid, nthreads, &S, &none, in, out, work->redOpArg, 0, 0, 0, work);
          down.recv(off(hi), len(hi));
        }
        {
          Prims ring(tid, nthreads, &prev, &next, in, out, work->redOpArg, 0, 0, 0, work);
          // S4 allgather from the owned section
          ring.sendFromOutput(off(hi), len(hi));
          for (int j = 1; j < nh - 1; ++j) ring.recvCopySend(off(hi - j), len(hi - j));
          ring.recv(off(hi + 1), len(hi + 1));
        }
        if (ordering == 2) {  // S2 last: global sum to the straggler, then our raw block
          {
            Prims up(tid, nthreads, &none, &S, in, out, work->redOpArg, 0, 0, 0, work);
            up.sendFromOutput(off(hi), len(hi));
          }
          Prims blk(tid, nthreads, &none, &S, in, out, work->redOpArg, 0, 0, 1, work);
          blk.send(blkOff, blkN);
        }
        if (!first) {  // the previous block's sum
          Prims link(tid, nthreads, &S, &none, in, out, work->redOpArg, 0, 1, 0, work);
          link.recv(prevOff, prevN);
        }
      } else {
        // ---- straggler: the sections first, one flow at a time in healthy order
        // (the arbiter's order); then the previous block's sum, ready since the
        // last loop, back to each healthy rank; then this loop's blocks, which
        // are needed only a loop later, while the straggler waits for the next
        // sections. Sections on connIndex 0, block traffic on connIndex 1.
        if (ordering == 1) {
          for (int i = 0; i < nh; ++i) {  // S2 + S3: partial of section i in, + our input, result back
            int h = oc->healthy[i];
            Prims p(tid, nthreads, &h, &h, in, out, work->redOpArg, 0, 0, 0, work);
            p.recvReduceCopySend(off(i), off(i), len(i));
          }
          if (!first) sendSums();
        } else {
          for (int i = 0; i < nh; ++i) {  // S3: raw section i to the rank that starts its chain
            int starter = oc->healthy[mod(i + 1)];
            Prims p(tid, nthreads, &none, &starter, in, out, work->redOpArg, 0, 0, 0, work);
            p.send(off(i), len(i));
          }
          if (!first) sendSums();
          for (int i = 0; i < nh; ++i) {  // S2: global sum of section i from its owner
            int h = oc->healthy[i];
            Prims p(tid, nthreads, &h, &none, in, out, work->redOpArg, 0, 0, 0, work);
            p.recv(off(i), len(i));
          }
        }
        // Each healthy rank's raw block added into ours: the first from the
        // input, the rest onto the running sum in the output.
        for (int i = 0; i < nh; ++i) {
          int h = oc->healthy[i];
          Prims q(tid, nthreads, &h, &none, i == 0 ? in : out, out, work->redOpArg, 0, 1, 0, work);
          q.recvReduceCopy(blkOff, blkOff, blkN);
        }
      }
      prevOff = blkOff;
      prevN = blkN;
    }
    // Flush: the last block's sum.
    if (hi >= 0) {
      Prims link(tid, nthreads, &S, &none, in, out, work->redOpArg, 0, 1, 0, work);
      link.recv(prevOff, prevN);
    } else {
      sendSums();
    }
  }
};
