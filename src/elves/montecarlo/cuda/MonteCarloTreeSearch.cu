#include "MonteCarloTreeSearch.h"

#include "common/Utils.h"

#include <cuda-utils/ErrorHandling.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

const size_t BLOCK_SIZE = 1;

__global__ void setupKernel(curandState* state, unsigned long seed);
__global__ void computeKernel(curandState* deviceStates, size_t reiterations, size_t dimension, Field* playfield, size_t* moveX, size_t* moveY, size_t* wins, size_t* visits);

void compute(size_t reiterations, size_t dimension, Field* playfield, size_t* moveX, size_t* moveY, size_t* wins, size_t* visits)
{
    dim3 dimGrid(div_ceil(dimension, BLOCK_SIZE), div_ceil(dimension, BLOCK_SIZE));
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    curandState* deviceStates;
    cudaMalloc(&deviceStates, sizeof(curandState));
    setupKernel <<< dimGrid, dimBlock >>>(deviceStates, 0ULL);
    computeKernel <<< dimGrid, dimBlock >>>(deviceStates, reiterations, dimension, playfield, moveX, moveY, wins, visits);
    CudaUtils::checkState();
}