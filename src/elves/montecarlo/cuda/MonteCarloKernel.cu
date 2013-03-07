#include "MonteCarloTreeSearch.h"
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include "OthelloField.h"
#include "CudaGameState.cuh"
#include "CudaMove.cuh"
#include "CudaSimulator.cuh"
#include "CudaUtil.cuh"
#include <assert.h>

__global__ void setupStateForRandom(curandState* state, size_t* seeds)
{
    printf("Block %d: Seed: %lu\n", blockIdx.x, seeds[blockIdx.x]);
	curand_init(seeds[blockIdx.x], 0, 0, &state[blockIdx.x]);
    //curand_init(0, 0, 0, &state[threadIdx.x]);
}


__device__ bool doStep(CudaGameState& state, CudaSimulator& simulator, size_t limit, float fakedRandom = -1)
{
    cassert(state.size == FIELD_DIMENSION * FIELD_DIMENSION, "Block %d, Thread %d detected invalid field size of %li\n", blockIdx.x, threadIdx.x, state.size);
    
    __syncthreads();
    
    simulator.calculatePossibleMoves();
    
    __syncthreads();
    
    size_t moveCount = simulator.countPossibleMoves();
    
    if (moveCount > 0)
    {
        size_t index = simulator.getRandomMoveIndex(moveCount, fakedRandom);
        cassert(index < state.size, "Block %d, Thread %d: Round %d detected unexpected move index %d for maximal playfield size %lu\n", blockIdx.x, limit, index, state.size);

        __syncthreads();

        simulator.flipEnemyCounter(index, limit);

        __syncthreads();
        
        cassert(!unchangedState(state, limit), "Block %d: %lu detected unchanged state\n", blockIdx.x, limit);
    }

    state.currentPlayer = state.getEnemyPlayer();
    return moveCount > 0;
}

const int MAXIMAL_MOVE_COUNT = 128; // an impossible condition - it would mean that for every field both players had to pass

__device__ void simulateGameLeaf(curandState* deviceState, CudaSimulator& simulator, CudaGameState& state, size_t* wins, size_t* visits)
{
    Player startingPlayer = state.currentPlayer;
    size_t passCounter = 0;
    size_t rounds = 0;

    __syncthreads();
    
    while (passCounter < 2)
    {
        bool passedMove = !doStep(state, simulator, rounds);
        passCounter = (passedMove ? passCounter + 1 : 0);

        cassert (rounds < MAXIMAL_MOVE_COUNT, "Detected rounds overflowing maximal count %d in %d\n", MAXIMAL_MOVE_COUNT, threadIdx.x); 
        rounds++;
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        ++(*visits);
        if (state.isWinner(startingPlayer))
        {
            ++(*wins);
        }
    }
}

__global__ void simulateGameLeaf(curandState* deviceState, Field* playfield, Player currentPlayer, size_t* wins, size_t* visits)
{
    int playfieldIndex = threadIdx.x;

    __shared__ Field sharedPlayfield[FIELD_DIMENSION * FIELD_DIMENSION];
    __shared__ Field oldPlayfield[FIELD_DIMENSION * FIELD_DIMENSION];
    __shared__ bool possibleMoves[FIELD_DIMENSION*FIELD_DIMENSION];
    sharedPlayfield[playfieldIndex] = playfield[playfieldIndex];

    CudaGameState state =  { 
        sharedPlayfield,
        oldPlayfield, 
        possibleMoves, 
        FIELD_DIMENSION * FIELD_DIMENSION, 
        FIELD_DIMENSION, 
        currentPlayer 
    };
    CudaSimulator simulator(&state, deviceState);
    simulateGameLeaf(deviceState, simulator, state, wins, visits);
}

__global__ void simulateGame(size_t reiterations, curandState* deviceStates, size_t numberOfPlayfields, Field* playfields, Player currentPlayer, OthelloResult* results)
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

        CudaGameState state =  { 
            sharedPlayfield, 
            oldPlayfield,
            possibleMoves, 
            FIELD_DIMENSION * FIELD_DIMENSION, 
            FIELD_DIMENSION, 
            currentPlayer 
        };
        CudaSimulator simulator(&state, deviceStates);

        size_t wins = 0;
        size_t visits = 0;

        __syncthreads();

        simulateGameLeaf(deviceStates, simulator, state, &wins, &visits);
        
        __syncthreads();
        if (threadIdx.x == 0)
        {
            results[node].wins += wins;
            results[node].visits += visits;
        }
    }
}

__global__ void testDoStep(curandState* deviceState, Field* playfield, Player currentPlayer, float fakedRandom)
{
    int playfieldIndex = threadIdx.x;
    __shared__ Field sharedPlayfield[FIELD_DIMENSION * FIELD_DIMENSION];
    __shared__ Field oldPlayfield[FIELD_DIMENSION * FIELD_DIMENSION];
    __shared__ bool possibleMoves[FIELD_DIMENSION*FIELD_DIMENSION];
    sharedPlayfield[playfieldIndex] = playfield[playfieldIndex];

    CudaGameState state =  { 
        sharedPlayfield, 
        oldPlayfield,
        possibleMoves, 
        FIELD_DIMENSION * FIELD_DIMENSION, 
        FIELD_DIMENSION, 
        currentPlayer 
    };
    CudaSimulator simulator(&state, deviceState);

    doStep(state, simulator, 0, fakedRandom);

    playfield[playfieldIndex] = sharedPlayfield[playfieldIndex];
}

__global__ void testSimulateGameLeaf(curandState* deviceState, Field* playfield, Player currentPlayer, size_t* wins, size_t* visits)
{
    int playfieldIndex = threadIdx.x;

    __shared__ Field sharedPlayfield[FIELD_DIMENSION * FIELD_DIMENSION];
    __shared__ Field oldPlayfield[FIELD_DIMENSION * FIELD_DIMENSION];
    __shared__ bool possibleMoves[FIELD_DIMENSION*FIELD_DIMENSION];
    sharedPlayfield[playfieldIndex] = playfield[playfieldIndex];

    CudaGameState state =  { 
        sharedPlayfield, 
        oldPlayfield,
        possibleMoves, 
        FIELD_DIMENSION * FIELD_DIMENSION, 
        FIELD_DIMENSION, 
        currentPlayer 
    };
    CudaSimulator simulator(&state, deviceState);
    simulateGameLeaf(deviceState, simulator, state, wins, visits);

	playfield[playfieldIndex] = sharedPlayfield[playfieldIndex];
}