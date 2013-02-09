#include "MonteCarloTreeSearch.h"

#include "common/Utils.h"

#include <cuda-utils/ErrorHandling.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

const int NUMBER_OF_BLOCKS = 1;
const int THREADS_PER_BLOCK = 64;

__global__ void setupStateForRandom(curandState* state, unsigned long seed);
__global__ void simulateGameLeaf(curandState* deviceState, Field* playfield, Player currentPlayer, size_t* wins, size_t* visits);
__global__ void simulateGame(curandState* deviceStates, size_t fieldDimension, Field* playfields, Player currentPlayer, size_t* wins, size_t* visits);

__global__ void testDoStep(curandState* deviceState, Field* playfield, Player currentPlayer, float fakedRandom);
__global__ void testSimulateGameLeaf(curandState* deviceState, Field* playfield, Player currentPlayer, size_t* wins, size_t* visits);

void leafSimulation(size_t reiterations, size_t dimension, Field* playfield, Player currentPlayer, size_t* moveX, size_t* moveY, size_t* wins, size_t* visits)
{
    curandState* deviceStates;
    cudaMalloc(&deviceStates, sizeof(curandState));
    setupStateForRandom <<< NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>(deviceStates, 0ULL);
    simulateGameLeaf <<< NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>(deviceStates, playfield, currentPlayer, wins, visits);
    CudaUtils::checkState();
}

void testBySimulateSingeStep(Field* playfield, Player currentPlayer, float fakedRandom)
{
    curandState* deviceStates;
    cudaMalloc(&deviceStates, sizeof(curandState));
    setupStateForRandom <<< NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>(deviceStates, 0ULL);
    testDoStep <<< NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>(deviceStates, playfield, currentPlayer, fakedRandom);
    CudaUtils::checkState();    
}

void testByLeafSimulation(size_t dimension, Field* playfield, Player currentPlayer, size_t* wins, size_t* visits)
{
    curandState* deviceStates;
    cudaMalloc(&deviceStates, sizeof(curandState));
    setupStateForRandom <<< NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>(deviceStates, 0ULL);
    testSimulateGameLeaf <<< NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>>(deviceStates, playfield, currentPlayer, wins, visits);
    CudaUtils::checkState();
}