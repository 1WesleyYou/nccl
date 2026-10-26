/*************************************************************************
 * OptCC AllReduce with bubble filling, for l < 2 - 1/(p-1) (paper Sec. 4.2,
 * Fig. 7, App. C). Included from all_reduce_optcc.h, which runs it when
 * NCCL_OPTCC_BF_NUM > 0.
 *
 * The schedule is OptccRingOrig's (all_reduce_optcc.h): four stages per
 * segment, ordering 1 on even channels and ordering 2 on odd ones. On top of
 * it, a point-to-point AllReduce of one block per segment through the
 * straggler. Every straggler <-> healthy transfer is one fused chunk on the
 * filling connection (connIndex 1, ncclTransportOptccConnect, its buffer
 * sized for a section plus a block): the block first, then the section
 * (Primitives::fusedOp), so a link carries one chunk per loop and direction,
 * back to back, and the block (ready early) fills the wait for the section.
 * connIndex 0 is left to the healthy ring.
 *
 * Chunks, loop j (blk_j = this loop's block, sum(blk_m) = its AllReduce):
 *   ordering 1, up   h_i -> S: [blk_j raw         | partial of section i]
 *               down S -> h_i: [sum(blk_{j-1})    | section i's result  ]
 *               flush (after the last loop L-1), down: [sum(blk_{L-1}) | -]
 *   ordering 2, down S -> starter of section i (healthy[i+1]), sent a loop
 *               ahead (chunk m carries loop m's raw section, sent in loop m-1):
 *                                [sum(blk_{m-2})   | S's raw section i  ]
 *               up   h_i -> S: [blk_j raw         | section i's global sum]
 *               flush, down: chunks m = L and L+1: [sum(blk_{m-2}) | -]
 * The straggler adds the healthy ranks' raw blocks into its own as they
 * arrive (healthy order; the first onto its input, the rest onto the running
 * sum in the output). An empty segment (loop 0 has no previous block sum) is
 * just absent: both ends compute the same layout.
 *
 * A loop is nh full sections (one ring chunk each) plus a block of r chunks
 * (r = NUM / DEN <= 1, ncclOptccBlkCount; the host's loop count uses the same
 * size). A short last loop shares out what is left: equal sections, and a
 * block only past nh full ones.
 *
 * Chunks per connection and direction, matching ncclProxySaveOp's
 * ncclPatternOptcc case (L = loops): ring as OptccRingOrig (connIndex 0); on
 * connIndex 1, up L, down L + 1 (ordering 1) or L + 2 (ordering 2). A
 * primitive call moves its connections by one chunk even when empty.
 ************************************************************************/

template <typename T, typename RedOp>
struct OptccRingBf {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS / ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>;
    // FanAsymmetric: send-only / recv-only stages pass -1 for the other side,
    // and FanSymmetric would take the recv count for both.
    using Prims = Primitives<T, RedOp, FanAsymmetric<1, 1>, 1, Proto, 0>;
    constexpr int In = 0, Out = 1;  // ncclFusedSeg buffers
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
    const int nLoops = (int)divUp(channelCount, loopCount);
    auto mod = [&] __device__(int j) -> int { return ((j % nh) + nh) % nh; };

    // Loop m's layout: its section size (a short last loop shares out what is
    // left; sections <= a chunk, since chunkCount is a multiple of align),
    // section i's offset and length (<= 0 on an empty tail), and its block.
    auto secOf = [&] __device__(int m) -> ssize_t {
      const ssize_t rem = channelCount - (ssize_t)m * loopCount;
      return rem < (ssize_t)nh * sec0 ? alignUp(divUp(rem, nh), align) : sec0;
    };
    auto off = [&] __device__(int m, int i) -> ssize_t {
      return gridOffset + (ssize_t)m * loopCount + (ssize_t)mod(i) * secOf(m);
    };
    auto len = [&] __device__(int m, int i) -> int {
      return (int)max((ssize_t)0, min(secOf(m), channelCount - (ssize_t)m * loopCount - (ssize_t)mod(i) * secOf(m)));
    };
    auto blkOff = [&] __device__(int m) -> ssize_t {
      return gridOffset + (ssize_t)m * loopCount + (ssize_t)nh * secOf(m);
    };
    auto blkN = [&] __device__(int m) -> int {  // 0 for loops that do not exist
      if (m < 0 || m >= nLoops) return 0;
      return (int)max((ssize_t)0, min(blkMax, channelCount - (ssize_t)m * loopCount - (ssize_t)nh * secOf(m)));
    };
    // Segments: nothing; sent from user memory; received (+ src) into the output.
    const ncclFusedSeg nil = {0, 0, -1, -1, 0, 0, 0};
    auto sendSeg = [&] __device__(int n, int buf, ssize_t ix) -> ncclFusedSeg { return {n, 0, -1, -1, 2, ix, buf}; };
    auto recvSeg = [&] __device__(int n, ssize_t dst, int srcBuf, ssize_t src) -> ncclFusedSeg {
      return {n, src, srcBuf, dst, 0, 0, 0};
    };
    // The straggler's own block of loop m plus the raw block of healthy i:
    // onto its input for the first healthy rank, onto the running sum after.
    auto addBlock = [&] __device__(int m, int i) -> ncclFusedSeg {
      return recvSeg(blkN(m), blkOff(m), i == 0 ? In : Out, blkOff(m));
    };

    if (hi >= 0) {
      // ---------------- healthy rank ----------------
      for (int m = 0; m < nLoops; ++m) {
        if (ordering == 2) {  // S3 + first hop of S1: S's raw section (hi-1) + ours -> next
          {
            Prims down(tid, nthreads, &S, &none, in, out, work->redOpArg, 0, 1, 0, work);
            down.fusedRecv(recvSeg(blkN(m - 2), blkOff(m - 2), -1, 0),
                           recvSeg(len(m, hi - 1), off(m, hi - 1), In, off(m, hi - 1)));
          }
          Prims ring(tid, nthreads, &prev, &next, in, out, work->redOpArg, 0, 0, 0, work);
          ring.sendFromOutput(off(m, hi - 1), len(m, hi - 1));
        }
        {
          Prims ring(tid, nthreads, &prev, &next, in, out, work->redOpArg, 0, 0, 0, work);
          // S1 reduce-scatter (runRing's steps with nranks = nh, ringIx = hi)
          if (ordering == 1) ring.send(off(m, hi - 1), len(m, hi - 1));
          for (int j = 2; j < nh; ++j) ring.recvReduceSend(off(m, hi - j), len(m, hi - j));
          ring.recvReduceCopy(off(m, hi), off(m, hi), len(m, hi));  // ordering 2: already the global sum
        }
        if (ordering == 1) {  // S2 + S3: [raw block | partial] up, [previous block's sum | result] back
          {
            Prims up(tid, nthreads, &none, &S, in, out, work->redOpArg, 0, 0, 1, work);
            up.fusedSend(sendSeg(blkN(m), In, blkOff(m)), sendSeg(len(m, hi), Out, off(m, hi)));
          }
          Prims down(tid, nthreads, &S, &none, in, out, work->redOpArg, 0, 1, 0, work);
          down.fusedRecv(recvSeg(blkN(m - 1), blkOff(m - 1), -1, 0), recvSeg(len(m, hi), off(m, hi), -1, 0));
        }
        {
          Prims ring(tid, nthreads, &prev, &next, in, out, work->redOpArg, 0, 0, 0, work);
          // S4 allgather from the owned section
          ring.sendFromOutput(off(m, hi), len(m, hi));
          for (int j = 1; j < nh - 1; ++j) ring.recvCopySend(off(m, hi - j), len(m, hi - j));
          ring.recv(off(m, hi + 1), len(m, hi + 1));
        }
        if (ordering == 2) {  // S2 last: [raw block | global sum] to the straggler
          Prims up(tid, nthreads, &none, &S, in, out, work->redOpArg, 0, 0, 1, work);
          up.fusedSend(sendSeg(blkN(m), In, blkOff(m)), sendSeg(len(m, hi), Out, off(m, hi)));
        }
      }
      // Flush: the block sums still on their way (ordering 1: the last one;
      // ordering 2: the last two).
      Prims down(tid, nthreads, &S, &none, in, out, work->redOpArg, 0, 1, 0, work);
      if (ordering == 2) down.fusedRecv(recvSeg(blkN(nLoops - 2), blkOff(nLoops - 2), -1, 0), nil);
      down.fusedRecv(recvSeg(blkN(nLoops - 1), blkOff(nLoops - 1), -1, 0), nil);
    } else if (ordering == 1) {
      // ---------------- straggler, ordering 1: one exchange per healthy rank,
      // in healthy order (the arbiter's): [raw block | partial] in, its block
      // added to ours and the partial plus our input into the output, and
      // [previous block's sum | result] back.
      for (int m = 0; m < nLoops; ++m) {
        for (int i = 0; i < nh; ++i) {
          int h = oc->healthy[i];
          const ncclFusedSeg sec = {len(m, i), off(m, i), In, off(m, i), 1, 0, 0};
          if (blkN(m - 1) == blkN(m)) {  // same layout both ways: one call, sends as it receives
            ncclFusedSeg blk = addBlock(m, i);
            blk.send = 2;
            blk.sendIx = blkOff(m - 1);
            blk.sendBuf = Out;
            Prims p(tid, nthreads, &h, &h, in, out, work->redOpArg, 0, 1, 1, work);
            p.fusedRecvSend(blk, sec);
          } else {  // loop 0 and a short last loop: receive, then send
            {
              Prims p(tid, nthreads, &h, &none, in, out, work->redOpArg, 0, 1, 0, work);
              p.fusedRecv(addBlock(m, i), recvSeg(len(m, i), off(m, i), In, off(m, i)));
            }
            Prims q(tid, nthreads, &none, &h, in, out, work->redOpArg, 0, 0, 1, work);
            q.fusedSend(sendSeg(blkN(m - 1), Out, blkOff(m - 1)), sendSeg(len(m, i), Out, off(m, i)));
          }
        }
      }
      for (int i = 0; i < nh; ++i) {  // flush: the last block's sum
        int h = oc->healthy[i];
        Prims q(tid, nthreads, &none, &h, in, out, work->redOpArg, 0, 0, 1, work);
        q.fusedSend(sendSeg(blkN(nLoops - 1), Out, blkOff(nLoops - 1)), nil);
      }
    } else {
      // ---------------- straggler, ordering 2: chunk m = [sum(blk_{m-2}) |
      // raw section of loop m] to the rank that starts section i's chain,
      // sent a loop ahead (they are input data); then [raw block | global sum]
      // from each healthy rank, its block added to ours.
      auto sendDown = [&] __device__(int m) {
        for (int i = 0; i < nh; ++i) {
          int starter = oc->healthy[mod(i + 1)];
          Prims q(tid, nthreads, &none, &starter, in, out, work->redOpArg, 0, 0, 1, work);
          q.fusedSend(sendSeg(blkN(m - 2), Out, blkOff(m - 2)), m < nLoops ? sendSeg(len(m, i), In, off(m, i)) : nil);
        }
      };
      if (nLoops > 0) sendDown(0);
      for (int m = 0; m < nLoops; ++m) {
        if (m + 1 < nLoops) sendDown(m + 1);
        for (int i = 0; i < nh; ++i) {
          int h = oc->healthy[i];
          Prims p(tid, nthreads, &h, &none, in, out, work->redOpArg, 0, 1, 0, work);
          p.fusedRecv(addBlock(m, i), recvSeg(len(m, i), off(m, i), -1, 0));
        }
      }
      sendDown(nLoops);      // flush: sum(blk_{L-2})
      sendDown(nLoops + 1);  //        sum(blk_{L-1})
    }
  }
};
