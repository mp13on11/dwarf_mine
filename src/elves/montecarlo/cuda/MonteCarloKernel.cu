#include "MonteCarloTreeSearch.h"
#include <curand.h>
#include <curand_kernel.h>
#include "OthelloField.h"
#include "CudaGameState.cuh"
#include "CudaMove.cuh"
#include "CudaSimulator.cuh"
#include "CudaUtil.cuh"
#include "CudaDebug.cuh"
#include <assert.h>

__global__ void setupStateForRandom(curandState* state, size_t* seeds)
{
	curand_init(seeds[blockIdx.x], 0, 0, &state[blockIdx.x]);
}


__device__ bool doStep(CudaGameState& state, CudaSimulator& simulator, size_t limit, float fakedRandom = -1)
{
    cassert(state.size == FIELD_DIMENSION * FIELD_DIMENSION, "Block %d, Thread %d detected invalid field size of %li\n", blockIdx.x, threadIdx.x, state.size);
    
    simulator.calculatePossibleMoves();
    
    size_t moveCount = simulator.countPossibleMoves();
    
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

__device__ void expandLeaf(curandState* deviceState, CudaSimulator& simulator, CudaGameState& state, size_t* wins, size_t* visits)
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

    if (threadIdx.x == 0) ++(*visits);
    if (state.isWinner(startingPlayer))
    {
        if (threadIdx.x == 0) ++(*wins);
    }
    
}

__global__ void simulateGame(size_t reiterations, curandState* deviceStates, size_t numberOfPlayfields, const Field* playfields, Player currentPlayer, OthelloResult* results)
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

        expandLeaf(deviceStates, simulator, state, &wins, &visits);
        
        __syncthreads();
        if (threadIdx.x == 0)
        {
            results[node].wins += wins;
            results[node].visits += visits;
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


__global__ void testDoStep(curandState* deviceState, Field* playfield, Player currentPlayer, float fakedRandom)
{
    int playfieldIndex = threadIdx.x;
    __shared__ Field sharedPlayfield[FIELD_DIMENSION * FIELD_DIMENSION];
    __shared__ Field oldPlayfield[FIELD_DIMENSION * FIELD_DIMENSION];
    __shared__ bool possibleMoves[FIELD_DIMENSION*FIELD_DIMENSION];
    sharedPlayfield[playfieldIndex] = playfield[playfieldIndex];

    // this part may be a shared variable?
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

__global__ void testExpandLeaf(curandState* deviceState, Field* playfield, Player currentPlayer, size_t* wins, size_t* visits)
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
    expandLeaf(deviceState, simulator, state, wins, visits);

	playfield[playfieldIndex] = sharedPlayfield[playfieldIndex];
}