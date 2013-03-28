#include "CudaProxy.h"
#include <curand.h>
#include <curand_kernel.h>
#include "Field.h"
#include "Move.h"
#include "State.cuh"
#include "Simulator.cuh"
#include "Random.cuh"
#include "Debug.cuh"
#include <assert.h>

__global__ void setupStateForRandom(curandState* state, size_t* seeds)
{
	curand_init(seeds[blockIdx.x], 0, 0, &state[blockIdx.x]);
}

__global__ void setupStateForRandom(curandState* states, float* randomValues, size_t numberOfRandomValues, size_t streamSeed = 0)
{
    curand_init(threadIdx.x + streamSeed, 0, 0, &states[threadIdx.x]);
    for (size_t i = 0; i + threadIdx.x < numberOfRandomValues; i += 128)
    {
        curandState deviceState = states[threadIdx.x];
        randomValues[i + threadIdx.x] = 1.0f - curand_uniform(&deviceState); // delivers (0, 1] - we need [0, 1)
        states[threadIdx.x] = deviceState;
    }
}

__global__ void simulateGame(size_t reiterations, curandState* deviceStates, size_t numberOfPlayfields, const Field* playfields, Player currentPlayer, Result* results)
{
    int playfieldIndex = threadIdx.x;

    for (size_t i = 0; i < reiterations; ++i)
    {
        size_t node = randomNumber(deviceStates, numberOfPlayfields);

        __shared__ Field sharedPlayfield[FIELD_DIMENSION * FIELD_DIMENSION];
        __shared__ Field oldPlayfield[FIELD_DIMENSION * FIELD_DIMENSION];
        __shared__ bool possibleMoves[FIELD_DIMENSION*FIELD_DIMENSION];
        
        size_t playfieldOffset = FIELD_DIMENSION * FIELD_DIMENSION * node;
        sharedPlayfield[playfieldIndex] = playfields[playfieldOffset + playfieldIndex];

        State state( 
            sharedPlayfield, 
            oldPlayfield,
            possibleMoves, 
            FIELD_DIMENSION, 
            currentPlayer 
        );
        Simulator simulator(&state, deviceStates);

        __syncthreads();

        simulator.expandLeaf();
        
        __syncthreads();

        if (state.isWinner(currentPlayer))
        {
            if (threadIdx.x == 0) 
			{
                results[node].wins++;
			}
        }
        if (threadIdx.x == 0)
        {
            results[node].visits++;
        }
    }
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

        __shared__ Field sharedPlayfield[FIELD_DIMENSION * FIELD_DIMENSION];
        __shared__ Field oldPlayfield[FIELD_DIMENSION * FIELD_DIMENSION];
        __shared__ bool possibleMoves[FIELD_DIMENSION*FIELD_DIMENSION];
        
        size_t playfieldOffset = FIELD_DIMENSION * FIELD_DIMENSION * node;
        sharedPlayfield[playfieldIndex] = playfields[playfieldOffset + playfieldIndex];

        State state(
            sharedPlayfield, 
            oldPlayfield,
            possibleMoves, 
            FIELD_DIMENSION, 
            currentPlayer
        );

        Simulator simulator(&state, randomValues, randomSeed);

        __syncthreads();

        simulator.expandLeaf();
        
        __syncthreads();
        if (state.isWinner(currentPlayer))
        {
            if (threadIdx.x == 0) results[node].wins++;
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

__global__ void testNumberOfMarkedFields(size_t* resultSum, bool* playfield)
{
    Field* stub = NULL;
    State state(
        stub, 
        stub,
        playfield, 
        FIELD_DIMENSION, 
        White 
    );

    *resultSum = state.numberOfMarkedFields();
}


__global__ void testDoStep(curandState* deviceState, Field* playfield, Player currentPlayer, float fakedRandom)
{
    int playfieldIndex = threadIdx.x;
    __shared__ Field sharedPlayfield[FIELD_DIMENSION * FIELD_DIMENSION];
    __shared__ Field oldPlayfield[FIELD_DIMENSION * FIELD_DIMENSION];
    __shared__ bool possibleMoves[FIELD_DIMENSION*FIELD_DIMENSION];
    sharedPlayfield[playfieldIndex] = playfield[playfieldIndex];

    // this part may be a shared variable?
    State state(
        sharedPlayfield, 
        oldPlayfield,
        possibleMoves, 
        FIELD_DIMENSION, 
        currentPlayer 
    );
    Simulator simulator(&state, deviceState);

    simulator.doStep(fakedRandom);

    playfield[playfieldIndex] = sharedPlayfield[playfieldIndex];
}

__global__ void testExpandLeaf(curandState* deviceState, Field* playfield, Player currentPlayer, size_t* wins, size_t* visits)
{
    int playfieldIndex = threadIdx.x;

    __shared__ Field sharedPlayfield[FIELD_DIMENSION * FIELD_DIMENSION];
    __shared__ Field oldPlayfield[FIELD_DIMENSION * FIELD_DIMENSION];
    __shared__ bool possibleMoves[FIELD_DIMENSION*FIELD_DIMENSION];
    sharedPlayfield[playfieldIndex] = playfield[playfieldIndex];

    State state(
        sharedPlayfield, 
        oldPlayfield,
        possibleMoves, 
        FIELD_DIMENSION, 
        currentPlayer 
    );
    Simulator simulator(&state, deviceState);
    simulator.expandLeaf();

    if (state.isWinner(currentPlayer))
    {
        if (threadIdx.x == 0) ++(*wins);
    }
    if (threadIdx.x == 0) ++(*visits);

	playfield[playfieldIndex] = sharedPlayfield[playfieldIndex];
}
