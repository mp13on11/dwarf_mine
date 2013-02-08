#include "MonteCarloTreeSearch.h"

#include "common/Utils.h"

#include <cuda-utils/ErrorHandling.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

const int NUMBER_OF_BLOCKS = 1;
const int THREADS_PER_BLOCK = 64;

__global__ void setupStateForRandom(curandState* state, unsigned long seed);
__global__ void simulateSingleStep(curandState* deviceState, Field* playfield, Player currentPlayer, float fakedRandom);
__global__ void simulateGameLeaf(curandState* deviceState, Field* playfield, Player currentPlayer);

void testBySimulateSingeStep(Field* playfield, Player currentPlayer, float fakedRandom)
{
	curandState* deviceStates;
    // cudaMalloc(&deviceStates, sizeof(curandState));
//    setupStateForRandom <<< NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>(deviceStates, 0ULL);
    //computeKernel <<< dimGrid, dimBlock >>>(deviceStates, reiterations, dimension, playfield, moveX, moveY, wins, visits);
    simulateSingleStep <<< NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>(deviceStates, playfield, currentPlayer, fakedRandom);
    CudaUtils::checkState();	
}


void compute(size_t reiterations, size_t dimension, Field* playfield, size_t* moveX, size_t* moveY, size_t* wins, size_t* visits)
{
    curandState* deviceStates;
    cudaMalloc(&deviceStates, sizeof(curandState));
    setupStateForRandom <<< NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>(deviceStates, 0ULL);
    //computeKernel <<< dimGrid, dimBlock >>>(deviceStates, reiterations, dimension, playfield, moveX, moveY, wins, visits);
    simulateGameLeaf <<< NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>(deviceStates, playfield, White);
    CudaUtils::checkState();
}