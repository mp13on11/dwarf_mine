#include "MonteCarloTreeSearch.h"

#include "common/Utils.h"

#include <cuda-utils/ErrorHandling.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

const int NUMBER_OF_BLOCKS = 1;
const int THREADS_PER_BLOCK = 64;
__global__ void setupKernel(curandState* state, unsigned long seed);
__global__ void computeKernel(curandState* deviceStates, size_t reiterations, size_t dimension, Field* playfield, size_t* moveX, size_t* moveY, size_t* wins, size_t* visits);
__global__ void computeSingleMove(Field* playfield, size_t moveX, size_t moveY, int directionX, int directionY, Player currentPlayer);

void compute(size_t reiterations, size_t dimension, Field* playfield, size_t* moveX, size_t* moveY, size_t* wins, size_t* visits)
{
    curandState* deviceStates;
    cudaMalloc(&deviceStates, sizeof(curandState));
    setupKernel <<< NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>(deviceStates, 0ULL);
    //computeKernel <<< dimGrid, dimBlock >>>(deviceStates, reiterations, dimension, playfield, moveX, moveY, wins, visits);
    computeSingleMove <<< NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>( playfield, 1U, 1U, 1, 1, White);
    CudaUtils::checkState();
}