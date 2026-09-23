#ifndef NCCL_GRAPH_OPTCC_H_
#define NCCL_GRAPH_OPTCC_H_

#include "core.h"

struct ncclComm;

// Reuse the original rank order and skip stragglers when finding local neighbors.
ncclResult_t ncclBuildOptccRing(struct ncclComm* comm, int channel);

#endif
