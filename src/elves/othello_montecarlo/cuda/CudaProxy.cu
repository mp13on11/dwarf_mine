/*****************************************************************************
* Dwarf Mine - The 13-11 Benchmark
*
* Copyright (c) 2013 Bünger, Thomas; Kieschnick, Christian; Kusber,
* Michael; Lohse, Henning; Wuttke, Nikolai; Xylander, Oliver; Yao, Gary;
* Zimmermann, Florian
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*****************************************************************************/

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


__global__ void setupStateForRandom(curandState* states, float* randomValues, size_t numberOfRandomValues);
__global__ void simulateGamePreRandom(size_t reiterations, size_t numberOfBlocks, float* randomValues, size_t numberOfPlayfields, const Field* playfields, Player currentPlayer, Result* results);


__global__ void testNumberOfMarkedFields(size_t* sum, const bool* playfield);
__global__ void testRandomNumber(float fakedRandom, size_t maximum, size_t* randomNumberResult);
__global__ void testDoStep(Field* playfield, Player currentPlayer, float fakedRandom);
__global__ void testExpandLeaf(Field* playfield, Player currentPlayer, size_t* wins, size_t* visits);

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

void testDoStepProxy(Field* playfield, Player currentPlayer, float fakedRandom)
{
    testDoStep <<< 1, THREADS_PER_BLOCK >>>(playfield, currentPlayer, fakedRandom);
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

void testExpandLeafProxy(Field* playfield, Player currentPlayer, size_t* wins, size_t* visits)
{
    testExpandLeaf <<< 1, THREADS_PER_BLOCK >>>(playfield, currentPlayer, wins, visits);
    CudaUtils::checkState();
