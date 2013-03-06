#include "MonteCarloTreeSearch.h"

#include "common/Utils.h"

#include <cuda-utils/ErrorHandling.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <cassert>
#include <cstdio>

const int THREADS_PER_BLOCK = 64;

__global__ void setupStateForRandom(curandState* state, size_t* seeds);
__global__ void simulateGameLeaf(curandState* deviceState, Field* playfield, Player currentPlayer, size_t* wins, size_t* visits);
__global__ void simulateGame(size_t reiterations, curandState* deviceStates, size_t numberOfPlayfields, Field* playfields, Player currentPlayer, OthelloResult* results);

__global__ void testDoStep(curandState* deviceState, Field* playfield, Player currentPlayer, float fakedRandom);
__global__ void testSimulateGameLeaf(curandState* deviceState, Field* playfield, Player currentPlayer, size_t* wins, size_t* visits);

void gameSimulation(size_t numberOfBlocks, size_t iterations, size_t* seeds, size_t numberOfPlayfields, Field* playfields, Player currentPlayer, OthelloResult* results)
{
    curandState* deviceStates;
    cudaMalloc(&deviceStates, sizeof(curandState) * numberOfBlocks);
    
    setupStateForRandom <<< numberOfBlocks, 1 >>> (deviceStates, seeds);
    CudaUtils::checkState();
    
    simulateGame <<< numberOfBlocks, THREADS_PER_BLOCK >>> (size_t(ceil(iterations * 1.0 / numberOfBlocks)), deviceStates, numberOfPlayfields, playfields, currentPlayer, results);
    CudaUtils::checkState();
}

void leafSimulation(size_t reiterations, size_t dimension, Field* playfield, Player currentPlayer, size_t* moveX, size_t* moveY, size_t* wins, size_t* visits)
{
    curandState* deviceStates;
    cudaMalloc(&deviceStates, sizeof(curandState) * 1 * THREADS_PER_BLOCK);
    
    setupStateForRandom <<< 1, THREADS_PER_BLOCK >>>(deviceStates, 0ULL);
    CudaUtils::checkState();

    simulateGameLeaf <<< 1, THREADS_PER_BLOCK >>>(deviceStates, playfield, currentPlayer, wins, visits);
    CudaUtils::checkState();
}

void testBySimulateSingeStep(Field* playfield, Player currentPlayer, float fakedRandom)
{
    curandState* deviceStates;
    cudaMalloc(&deviceStates, sizeof(curandState) * 1 * THREADS_PER_BLOCK);
    setupStateForRandom <<< 1, THREADS_PER_BLOCK >>>(deviceStates, 0ULL);
    CudaUtils::checkState();

    testDoStep <<< 1, THREADS_PER_BLOCK >>>(deviceStates, playfield, currentPlayer, fakedRandom);
    CudaUtils::checkState();    
}

void testByLeafSimulation(size_t dimension, Field* playfield, Player currentPlayer, size_t* wins, size_t* visits)
{
    curandState* deviceStates;
    cudaMalloc(&deviceStates, sizeof(curandState) * 1 * THREADS_PER_BLOCK);
    setupStateForRandom <<< 1, THREADS_PER_BLOCK >>>(deviceStates, 0UL);
    CudaUtils::checkState();
    
    testSimulateGameLeaf <<< 1, THREADS_PER_BLOCK >>>(deviceStates, playfield, currentPlayer, wins, visits);
    CudaUtils::checkState();
}