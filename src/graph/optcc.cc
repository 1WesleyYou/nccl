#include "optcc.h"
#include "comm.h"

static bool isStraggler(const int* ranks, int count, int rank) {
  for (int i=0; i<count; i++) if (ranks[i] == rank) return true;
  return false;
}

ncclResult_t ncclBuildOptccRings(int nChannels, const int* rings, struct ncclComm* comm,
                                 int* stragglerRanks, int nStragglers) {
  const int nranks = comm->nRanks;
  const int nHealthyRanks = nranks-nStragglers;
  if (nHealthyRanks <= 0) return ncclInvalidArgument;
  const bool localIsStraggler = isStraggler(stragglerRanks, nStragglers, comm->rank);

  for (int c=0; c<nChannels; c++) {
    struct ncclOptccRing* optcc = &comm->channels[c].optccRing;
    optcc->nHealthyRanks = nHealthyRanks;
    optcc->nStragglers = nStragglers;
    optcc->stragglerRanks = stragglerRanks;
    optcc->ringPrev = optcc->ringNext = -1;

    // A straggler is outside the healthy ring.
    if (localIsStraggler) continue;

    // ncclBuildRings ordered each channel from the current rank; skip stragglers.
    const int* order = rings+c*nranks;
    int prev = nranks-1, next = 1%nranks;
    while (isStraggler(stragglerRanks, nStragglers, order[prev])) prev = (prev+nranks-1)%nranks;
    while (isStraggler(stragglerRanks, nStragglers, order[next])) next = (next+1)%nranks;
    optcc->ringPrev = order[prev];
    optcc->ringNext = order[next];
  }
  return ncclSuccess;
}
