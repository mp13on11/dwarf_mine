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
#include <curand.h>
#include <curand_kernel.h>
#include "Field.h"
#include "State.cuh"
#include "Move.h"
#include "Simulator.cuh"
#include "Random.cuh"
#include "Debug.cuh"
#include <assert.h>

__global__ void setupStateForRandom(curandState* states, float* randomValues, size_t numberOfRandomValues)
{
    curand_init(threadIdx.x, 0, 0, &states[threadIdx.x]);
    for (size_t i = 0; i + threadIdx.x < numberOfRandomValues; i += 128)
    {
        curandState deviceState = states[threadIdx.x];
        randomValues[i + threadIdx.x] = 1.0f - curand_uniform(&deviceState); // delivers (0, 1] - we need [0, 1)
        states[threadIdx.x] = deviceState;
    }
}

__device__ bool doStep(CudaGameState& state, CudaSimulator& simulator, size_t limit, float fakedRandom = -1)
{
    cassert(state.size == FIELD_DIMENSION * FIELD_DIMENSION, "Block %d, Thread %d detected invalid field size of %li\n", blockIdx.x, threadIdx.x, state.size);
    
    __syncthreads();

    simulator.calculatePossibleMoves();
    
    size_t moveCount = simulator.countPossibleMoves();

    __syncthreads();

    if (moveCount > 0)
    {
        size_t index = simulator.getRandomMoveIndex(moveCount, fakedRandom);
        cassert(index < state.size, "Block %d, Thread %d: Round %d detected unexpected move index %d for maximal playfield size %lu\n", blockIdx.x, limit, index, state.size);

        simulator.flipEnemyCounter(index, limit);

        cassert(!state.isUnchanged(), "Block %d: %lu detected unchanged state\n", blockIdx.x, limit);
    }

    state.currentPlayer = state.getEnemyPlayer();
    return moveCount > 0;
}

__device__ void expandLeaf(CudaSimulator& simulator, CudaGameState& state)
{
    size_t passCounter = 0;
    size_t rounds = 0;

    while (passCounter < 2)
    {
        bool passedMove = !doStep(state, simulator, rounds);
        passCounter = (passedMove ? passCounter + 1 : 0);

        cassert (rounds < MAXIMAL_MOVE_COUNT, "Detected rounds overflowing maximal count %d in %d\n", MAXIMAL_MOVE_COUNT, threadIdx.x); 
        rounds++;
    }
    __syncthreads();
}

__global__ void simulateGamePreRandom(size_t reiterations, size_t numberOfBlocks, float* randomValues, size_t numberOfPlayfields, const Field* playfields, Player currentPlayer, Result* results)
{
    int playfieldIndex = threadIdx.x;
    size_t blockIterations = size_t(ceil(reiterations * 1.0 / numberOfBlocks));

    for (size_t i = 0; i < blockIterations; ++i)
    {
		size_t randomSeed = i * numberOfBlocks + blockIdx.x;

        cassert(randomSeed < reiterations + 121, "SeedIndex %lu exceeded reiterations\n", randomSeed);
        size_t node = randomNumber(randomValues, &randomSeed, numberOfPlayfields);

        __shared__ Field sharedPlayfield[FIELD_SIZE];
        __shared__ Field oldPlayfield[FIELD_SIZE];
        __shared__ bool possibleMoves[FIELD_SIZE];
        
        size_t playfieldOffset = FIELD_SIZE * node;
        sharedPlayfield[playfieldIndex] = playfields[playfieldOffset + playfieldIndex];

        CudaGameState state(
            sharedPlayfield, 
            oldPlayfield,
            possibleMoves, 
            FIELD_DIMENSION, 
            currentPlayer 
        );

        CudaSimulator simulator(&state, randomValues, randomSeed);

        __syncthreads();

        expandLeaf(simulator, state);
        
        __syncthreads();
        if (state.isWinner(currentPlayer))
        {
            if (threadIdx.x == 0)
                results[node].wins++;
        }
        if (threadIdx.x == 0)
        {
            results[node].visits ++;
        }
    }
}

__global__ void testRandomNumber(float fakedRandom, size_t maximum, size_t* randomNumberResult)
{
    *randomNumberResult = randomNumber(NULL, maximum, fakedRandom);
}

__global__ void testNumberOfMarkedFields(size_t* resultSum, const bool* playfield)
{
    *resultSum = numberOfMarkedFields(playfield);
}


__global__ void testDoStep(Field* playfield, Player currentPlayer, float fakedRandom)
{
    int playfieldIndex = threadIdx.x;

    __shared__ Field sharedPlayfield[FIELD_SIZE];
    __shared__ Field oldPlayfield[FIELD_SIZE];
    __shared__ bool possibleMoves[FIELD_SIZE];

    sharedPlayfield[playfieldIndex] = playfield[playfieldIndex];

    // this part may be a shared variable?
    CudaGameState state(
        sharedPlayfield, 
        oldPlayfield,
        possibleMoves, 
        FIELD_DIMENSION, 
        currentPlayer 
    );

    CudaSimulator simulator(&state, &fakedRandom, 0);

    doStep(state, simulator, 0, fakedRandom);

    playfield[playfieldIndex] = sharedPlayfield[playfieldIndex];
}

__global__ void testExpandLeaf(Field* playfield, Player currentPlayer, size_t* wins, size_t* visits)
{
    int playfieldIndex = threadIdx.x;

    __shared__ Field sharedPlayfield[FIELD_SIZE];
    __shared__ Field oldPlayfield[FIELD_SIZE];
    __shared__ bool possibleMoves[FIELD_SIZE];

    sharedPlayfield[playfieldIndex] = playfield[playfieldIndex];

    CudaGameState state (
        sharedPlayfield, 
        oldPlayfield,
        possibleMoves, 
        FIELD_DIMENSION, 
        currentPlayer 
    );

    float fakedRandom = 0;
    CudaSimulator simulator(&state, &fakedRandom, 0);

    __syncthreads();

    expandLeaf(simulator, state);

    if (state.isWinner(currentPlayer))
    {
        if (threadIdx.x == 0) ++(*wins);
    }
    if (threadIdx.x == 0)
    {
        ++(*visits);
    }
    playfield[playfieldIndex] = sharedPlayfield[playfieldIndex];
}
