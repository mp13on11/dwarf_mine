#include "MonteCarloTreeSearch.h"

#include "common/Utils.h"

#include <cuda-utils/ErrorHandling.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <cassert>
#include <cstdio>

const int NUMBER_OF_BLOCKS = 1;
const int THREADS_PER_BLOCK = 64;

__global__ void setupStateForRandom(curandState* state, unsigned long seed);
__global__ void simulateGameLeaf(curandState* deviceState, Field* playfield, Player currentPlayer, size_t* wins, size_t* visits);
__global__ void simulateGame(size_t reiterations, curandState* deviceStates, size_t numberOfPlayfields, Field* playfields, Player currentPlayer, OthelloResult* results);

__global__ void testDoStep(curandState* deviceState, Field* playfield, Player currentPlayer, float fakedRandom);
__global__ void testSimulateGameLeaf(curandState* deviceState, Field* playfield, Player currentPlayer, size_t* wins, size_t* visits);

static size_t seed = 70;

void gameSimulation(size_t reiterations, size_t numberOfPlayfields, Field* playfields, Player currentPlayer, OthelloResult* results)
{
    curandState* deviceStates;
    cudaMalloc(&deviceStates, sizeof(curandState) * NUMBER_OF_BLOCKS);
    //size_t seed = time(NULL);
    std::cout<<"Seed: "<< seed << std::endl;
    setupStateForRandom <<< NUMBER_OF_BLOCKS, 1 >>> (deviceStates, seed);
    seed++;
    simulateGame <<< NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >>> (reiterations, deviceStates, numberOfPlayfields, playfields, currentPlayer, results);
    CudaUtils::checkState();
}

void leafSimulation(size_t reiterations, size_t dimension, Field* playfield, Player currentPlayer, size_t* moveX, size_t* moveY, size_t* wins, size_t* visits)
{
    curandState* deviceStates;
    cudaMalloc(&deviceStates, sizeof(curandState));
    setupStateForRandom <<< 1, THREADS_PER_BLOCK >>>(deviceStates, 0ULL);
    simulateGameLeaf <<< 1, THREADS_PER_BLOCK >>>(deviceStates, playfield, currentPlayer, wins, visits);
    CudaUtils::checkState();
}

void testBySimulateSingeStep(Field* playfield, Player currentPlayer, float fakedRandom)
{
    curandState* deviceStates;
    cudaMalloc(&deviceStates, sizeof(curandState));
    setupStateForRandom <<< 1, THREADS_PER_BLOCK >>>(deviceStates, 0ULL);
    testDoStep <<< 1, THREADS_PER_BLOCK >>>(deviceStates, playfield, currentPlayer, fakedRandom);
    CudaUtils::checkState();    
}

void testByLeafSimulation(size_t dimension, Field* playfield, Player currentPlayer, size_t* wins, size_t* visits)
{
    curandState* deviceStates;
    cudaMalloc(&deviceStates, sizeof(curandState));
    setupStateForRandom <<< 1, THREADS_PER_BLOCK >>>(deviceStates, 0UL);
    testSimulateGameLeaf <<< 1, THREADS_PER_BLOCK >>>(deviceStates, playfield, currentPlayer, wins, visits);
    CudaUtils::checkState();
}