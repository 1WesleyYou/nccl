/*************************************************************************
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// input the next[] node ring, return the rings that put every found node in order.
ncclResult_t ncclBuildRings(int nrings, int* rings, int rank, int nranks, int* prev, int* next);
