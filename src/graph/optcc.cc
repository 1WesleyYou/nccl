#include "optcc.h"
#include "comm.h"

static bool isStraggler(const int* ranks, int count, int rank) {
  for (int i=0; i<count; i++) if (ranks[i] == rank) return true;
  return false;
}

ncclResult_t ncclBuildOptccRing(struct ncclComm* comm, int channel) {
  struct ncclChannel* ch = &comm->channels[channel];
  struct ncclOptccRing* optcc = &ch->optccRing;
  // Count healthy ranks; the original ring still contains every rank.
  const int nranks = comm->nRanks;
  optcc->nHealthyRanks = nranks-optcc->nStragglers;
  if (optcc->nHealthyRanks <= 0) return ncclInvalidArgument;

  // Read the full ring order, rotated to start at this rank; never store or modify it.
  const int* order = ch->ring.userRanks;
  optcc->ringPrev = optcc->ringNext = -1;

  // if current rank is straggler: just leave the function (outside the ring)
  if (isStraggler(optcc->stragglerRanks, optcc->nStragglers, comm->rank)) return ncclSuccess;

  // Walk both directions to find the nearest healthy neighbors.
  int prev = nranks-1, next = 1%nranks;
  while (isStraggler(optcc->stragglerRanks, optcc->nStragglers, order[prev])) {
    prev = (prev+nranks-1)%nranks;
  }
  while (isStraggler(optcc->stragglerRanks, optcc->nStragglers, order[next])) {
    next = (next+1)%nranks;
  }
  optcc->ringPrev = order[prev];
  optcc->ringNext = order[next];
  return ncclSuccess;
}
