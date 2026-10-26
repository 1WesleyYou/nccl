/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "transport.h"
#include "bootstrap.h"

ncclResult_t ncclTransportRingConnect(struct ncclComm* comm) {
  struct ringConnInfo {
    bool useNetPXN;
    bool useGdr;
  };
  struct ringConnInfo* ringInfo = NULL;
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nRanks > 1) {
    comm->useGdr = true;
    comm->useNetPXN = false;
    for (int c = 0; c < comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->ring.prev, 1, &channel->ring.next, 0), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_RING], 0), ret, fail);
    if (ncclParamLocalRegister() || ncclParamGraphRegister()) {
      NCCLCHECK(ncclCalloc(&ringInfo, comm->nRanks));
      ringInfo[comm->rank].useGdr = comm->useGdr;
      ringInfo[comm->rank].useNetPXN = comm->useNetPXN;
      NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, ringInfo, sizeof(struct ringConnInfo)), ret, fail);
      for (int i = 0; i < comm->nRanks; ++i) {
        if (!ringInfo[i].useGdr) comm->useGdr = false;
        if (ringInfo[i].useNetPXN) comm->useNetPXN = true;
        if (comm->useGdr == false && comm->useNetPXN == true) break;
      }
    }
    INFO(NCCL_INIT, "Connected all rings, use ring PXN %d GDR %d", comm->useNetPXN, comm->useGdr);
  }
exit:
  free(ringInfo);
  return ret;
fail:
  goto exit;
}

static inline bool optccIsStraggler(const struct ncclOptccRing* optcc, int rank) {
  for (int i=0; i<optcc->nStragglers; i++) {
    if (optcc->stragglerRanks[i] == rank) return true;
  }
  return false;
}

// P2P connection is to establish a logical edge (from topo logic to conn logic)
// P2P setup is to really connect a physical edge base on conn info
ncclResult_t ncclTransportOptccConnect(struct ncclComm* comm) {
  struct optccConnInfo {
    bool useNetPXN;
    bool useGdr;
  };
  struct optccConnInfo* optccInfo = NULL;
  ncclResult_t ret = ncclSuccess;
  if (comm->nRanks <= 1) return ncclSuccess;
  struct ncclOptccRing* topology = &comm->channels[0].optccRing;

  // check if local rank is a straggler
  const bool isStraggler = optccIsStraggler(topology, comm->rank);

  for (int c=0; c<comm->nChannels; c++) {
    struct ncclOptccRing* optcc = &comm->channels[c].optccRing;
    if (!isStraggler) {
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &optcc->ringPrev, 1, &optcc->ringNext, 0), ret, fail);
      for (int i=0; i<optcc->nStragglers; i++) {
        int peer = optcc->stragglerRanks[i];
        NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &peer, 1, &peer, 0), ret, fail);
      }
    } else {
      // Each straggler must register its own send and recv endpoints too.
      for (int r=0; r<comm->nRanks; r++) {
        if (optccIsStraggler(optcc, r)) continue;
        NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &r, 1, &r, 0), ret, fail);
      }
    }
  }

  // The graph picks a NIC per channel, not per peer, so extra OptCC edges use the ring's NICs.
  NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_OPTCCRING], 0), ret, fail);

  // Bubble filling: the blocks and their sums get a second healthy <-> straggler
  // connection (connIndex 1), so they have their own buffers, FIFO and QPs and
  // never queue behind a section. Built here at init with the rest, not lazily.
  // Passing the graph gives it dedicated buffers of the collective size.
  if (topology->bfNum > 0) {
    // A chunk on these connections is a section s plus a block t = s * NUM / DEN
    // (all_reduce_optcc_bf.h), so their buffer is the ring's scaled by
    // 1 + NUM / DEN, with room for the slices' 16-element rounding.
    const int64_t ring = comm->buffSizes[NCCL_PROTO_SIMPLE];
    comm->optccBfBuffSize = (int)alignUp(ring + ring * topology->bfNum / topology->bfDen + 4096, (int64_t)4096);
    for (int c=0; c<comm->nChannels; c++) {
      struct ncclOptccRing* optcc = &comm->channels[c].optccRing;
      if (!isStraggler) {
        for (int i=0; i<optcc->nStragglers; i++) {
          int peer = optcc->stragglerRanks[i];
          NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &peer, 1, &peer, 1), ret, fail);
        }
      } else {
        for (int r=0; r<comm->nRanks; r++) {
          if (optccIsStraggler(optcc, r)) continue;
          NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &r, 1, &r, 1), ret, fail);
        }
      }
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_OPTCCRING], 1), ret, fail);
    // Only the NET transport sizes these buffers per connection (net.cc); a
    // P2P / SHM peer would get the comm's smaller buffer and overflow it.
    for (int c=0; c<comm->nChannels; c++) {
      for (int r=0; r<comm->nRanks; r++) {
        struct ncclChannelPeer* peer = comm->channels[c].peers[r];
        if (r == comm->rank || peer == NULL) continue;
        for (struct ncclConnector* conn : {peer->send + 1, peer->recv + 1}) {
          if (conn->connected && conn->conn.stepSize != comm->optccBfBuffSize / NCCL_STEPS) {
            WARN("OptCC bubble filling: channel %d peer %d connIndex 1 has step size %d, not %d; filling needs the NET transport between healthy ranks and the straggler",
                 c, r, conn->conn.stepSize, comm->optccBfBuffSize / NCCL_STEPS);
            ret = ncclInvalidUsage;
            goto fail;
          }
        }
      }
    }
  }
  // Unlike the ring, do not reset useGdr/useNetPXN: slots other algorithms already built are skipped above.
  if (ncclParamLocalRegister() || ncclParamGraphRegister()) {  // software config read
    NCCLCHECK(ncclCalloc(&optccInfo, comm->nRanks));
    optccInfo[comm->rank].useGdr = comm->useGdr;
    optccInfo[comm->rank].useNetPXN = comm->useNetPXN;
    NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, optccInfo, sizeof(struct optccConnInfo)), ret, fail);
    for (int i = 0; i < comm->nRanks; ++i) {
      if (!optccInfo[i].useGdr) comm->useGdr = false;
      if (optccInfo[i].useNetPXN) comm->useNetPXN = true;
      if (comm->useGdr == false && comm->useNetPXN == true) break;
    }
  }
  comm->initAlgoChannels[NCCL_ALGO_OPTCCRING] = true;
  INFO(NCCL_INIT, "Connected optcc rings, stragglers %d, PXN %d GDR %d, filling conns %d (buffer %d)",
       topology->nStragglers, comm->useNetPXN, comm->useGdr, topology->bfNum > 0, comm->optccBfBuffSize);
exit:
  free(optccInfo);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTransportTreeConnect(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nRanks > 1) {
    // Connect Trees
    for (int c = 0; c < comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_TREE_ARITY, channel->tree.down, 1, &channel->tree.up, 0), ret, fail);
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->tree.up, NCCL_MAX_TREE_ARITY, channel->tree.down, 0), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
    INFO(NCCL_INIT, "Connected all trees");
  }
exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTransportPatConnect(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nRanks > 1) {
    for (int mask=1; mask<comm->nRanks; mask<<=1) {
      int prevPeer = (comm->rank + mask) % comm->nRanks;
      int nextPeer = (comm->rank + comm->nRanks - mask) % comm->nRanks;
      for (int c = 0; c < comm->nChannels; c++) {
        NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &prevPeer, 1, &nextPeer, 0), ret, fail); // ReduceScatter
      }
      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
      for (int c = 0; c < comm->nChannels; c++) {
        NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &nextPeer, 1, &prevPeer, 0), ret, fail); // AllGather
      }
      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
    }
    INFO(NCCL_INIT, "Connected binomial trees");
  }
exit:
  return ret;
fail:
  goto exit;
}
