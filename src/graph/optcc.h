#ifndef NCCL_GRAPH_OPTCC_H_
#define NCCL_GRAPH_OPTCC_H_

#include "core.h"

struct ncclComm;

// Build each channel's healthy-ring neighbors from the original ring order.
ncclResult_t ncclBuildOptccRings(int nChannels, const int* rings, struct ncclComm* comm,
                                 int* stragglerRanks, int nStragglers);

#endif
