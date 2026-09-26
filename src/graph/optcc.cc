#include "optcc.h"
#include "comm.h"
#include "param.h"
#include <cctype>
#include <cstring>
#include <string>

NCCL_PARAM(OptccStraggler, "OPTCC_STRAGGLER", 2);
NCCL_PARAM(OptccStaggerNs, "OPTCC_STAGGER_NS", 0);
int ncclOptccStraggler() { return (int)ncclParamOptccStraggler(); }

bool ncclOptccRequested() {
  static int requested = -1;
  if (requested < 0) {
    const char* env = ncclGetEnv("NCCL_ALGO");
    std::string s = env ? env : "";
    for (auto& ch : s) ch = (char)std::tolower((unsigned char)ch);
    requested = (!s.empty() && s[0] != '^' && s.find("optccring") != std::string::npos) ? 1 : 0;
  }
  return requested == 1;
}

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
    optcc->straggler = nStragglers > 0 ? stragglerRanks[0] : -1;
    optcc->healthyIndex = -1;
    optcc->staggerNs = (int)ncclParamOptccStaggerNs();
    {
      // Canonical healthy order: this channel's ring, rotated to start at its lowest rank.
      const int* ord = rings+c*nranks;
      int start = 0;
      for (int i=1; i<nranks; i++) if (ord[i] < ord[start]) start = i;
      int n = 0;
      for (int i=0; i<nranks; i++) {
        int r = ord[(start+i)%nranks];
        if (isStraggler(stragglerRanks, nStragglers, r)) continue;
        if (n < NCCL_OPTCC_MAX_HEALTHY) optcc->healthy[n] = r;
        if (r == comm->rank) optcc->healthyIndex = n;
        n++;
      }
      if (n > NCCL_OPTCC_MAX_HEALTHY) return ncclInvalidUsage;
    }

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
