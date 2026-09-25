#ifndef NCCL_GRAPH_OPTCC_H_
#define NCCL_GRAPH_OPTCC_H_

#include "core.h"

struct ncclComm;

// True when NCCL_ALGO names OptccRing (not as a ^exclusion): OptCC runs only
// when asked for by name; otherwise NCCL behaves as upstream.
bool ncclOptccRequested();
// The straggler rank (NCCL_OPTCC_STRAGGLER, default 2).
int ncclOptccStraggler();

// Build each channel's healthy-ring neighbors from the original ring order.
ncclResult_t ncclBuildOptccRings(int nChannels, const int* rings, struct ncclComm* comm,
                                 int* stragglerRanks, int nStragglers);

#endif
