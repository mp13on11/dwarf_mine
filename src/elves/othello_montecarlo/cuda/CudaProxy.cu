#include "CudaProxy.h"

#include "common/Utils.h"

#include <cuda-utils/ErrorHandling.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <cassert>
#include <cstdio>

const int THREADS_PER_BLOCK = 64;
const int THREADS_PER_BLOCK_RANDOM_KERNEL = 128;


__global__ void setupStateForRandom(curandState* state, size_t* seeds);
__global__ void setupStateForRandom(curandState* states, float* randomValues, size_t numberOfRandomValues);
__global__ void simulateGamePreRandom(size_t reiterations, size_t numberOfBlocks, float* randomValues, size_t numberOfPlayfields, const Field* playfields, Player currentPlayer, Result* results);


__global__ void testNumberOfMarkedFields(size_t* sum, const bool* playfield);
__global__ void testRandomNumber(float fakedRandom, size_t maximum, size_t* randomNumberResult);
__global__ void testDoStep(curandState* deviceState, Field* playfield, Player currentPlayer, float fakedRandom);
__global__ void testExpandLeaf(curandState* deviceState, Field* playfield, Player currentPlayer, size_t* wins, size_t* visits);

void gameSimulationPreRandom(size_t numberOfBlocks, size_t iterations, float* randomValues, size_t numberOfRandomValues, size_t numberOfPlayfields, const Field* playfields, Player currentPlayer, Result* results)
{
    curandState* deviceStates;
    cudaMalloc(&deviceStates, sizeof(curandState) * THREADS_PER_BLOCK_RANDOM_KERNEL);
    setupStateForRandom<<< 1, THREADS_PER_BLOCK_RANDOM_KERNEL >>>(deviceStates, randomValues, numberOfRandomValues);
    CudaUtils::checkState();

    simulateGamePreRandom <<< numberOfBlocks, THREADS_PER_BLOCK >>> (iterations, numberOfBlocks, randomValues, numberOfPlayfields, playfields, currentPlayer, results);
    CudaUtils::checkState();
}

void gameSimulationPreRandomStreamed(size_t numberOfBlocks, size_t iterations, float* randomValues, size_t numberOfRandomValues, size_t numberOfPlayfields, const Field* playfields, Player currentPlayer, Result* results, cudaStream_t stream, size_t streamSeed)
{
    curandState* deviceStates;
    cudaMalloc(&deviceStates, sizeof(curandState) * THREADS_PER_BLOCK_RANDOM_KERNEL);

    setupStateForRandom<<< 1, THREADS_PER_BLOCK_RANDOM_KERNEL, 0, stream >>>(deviceStates, randomValues, numberOfRandomValues);
    CudaUtils::checkState();

    simulateGamePreRandom <<< numberOfBlocks, THREADS_PER_BLOCK, 0, stream >>> (iterations, numberOfBlocks, randomValues, numberOfPlayfields, playfields, currentPlayer, results);
    CudaUtils::checkState();
}

void setupSeedForTest(size_t numberOfBlocks, curandState* deviceStates)
{
    size_t* seed;
    
    cudaMalloc(&seed, sizeof(size_t) * numberOfBlocks);
    cudaMalloc(&deviceStates, sizeof(curandState) * numberOfBlocks);
    
    setupStateForRandom <<< numberOfBlocks, 1 >>>(deviceStates, seed);
    
    CudaUtils::checkState();
}

void testDoStepProxy(Field* playfield, Player currentPlayer, float fakedRandom)
{
    curandState* deviceStates = NULL;
    size_t numberOfBlocks = 1;
    setupSeedForTest(numberOfBlocks, deviceStates);

    testDoStep <<< numberOfBlocks, THREADS_PER_BLOCK >>>(deviceStates, playfield, currentPlayer, fakedRandom);
    CudaUtils::checkState();    
}

void testNumberOfMarkedFieldsProxy(size_t* sum, const bool* playfield)
{
    testNumberOfMarkedFields<<< 1, THREADS_PER_BLOCK >>>(sum, playfield);
    CudaUtils::checkState();
}

void testRandomNumberProxy(float fakedRandom, size_t maximum, size_t* randomMoveIndex)
{
    testRandomNumber<<< 1, 1 >>> (fakedRandom, maximum, randomMoveIndex);
    CudaUtils::checkState();
}

void testExpandLeafProxy(size_t dimension, Field* playfield, Player currentPlayer, size_t* wins, size_t* visits)
{
    curandState* deviceStates = NULL;
    size_t numberOfBlocks = 1;
    setupSeedForTest(numberOfBlocks, deviceStates);
    
    testExpandLeaf <<< numberOfBlocks, THREADS_PER_BLOCK >>>(deviceStates, playfield, currentPlayer, wins, visits);
    CudaUtils::checkState();
}
