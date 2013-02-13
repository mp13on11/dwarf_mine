#include "MonteCarloTreeSearch.h"
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include "OthelloField.h"
#include "CudaGameState.cuh"
#include "CudaMove.cuh"
#include "CudaSimulator.cuh"
#include "CudaUtil.cuh"


__global__ void setupStateForRandom(curandState* state, unsigned long seed)
{
	int id = 0; // threadIdx.x;
	curand_init(seed, id, 0, &state[id]);
}

__device__ bool doStep(CudaGameState& state, CudaSimulator& simulator, float fakedRandom = -1)
{
    __syncthreads();

    simulator.calculatePossibleMoves();
    size_t moveCount = simulator.countPossibleMoves();
    if (threadIdx.x == 0)
        printf("Block %d: Moves %lu \n", blockIdx.x, moveCount);
    if (moveCount > 0)
    {
        __shared__ size_t index;
        if (threadIdx.x == 0)
            index = simulator.getRandomMoveIndex(moveCount, fakedRandom);
        
        __syncthreads();

        simulator.flipEnemyCounter(index);

        __syncthreads();
        state.field[index] = state.currentPlayer;
    }
    state.currentPlayer = state.getEnemyPlayer();
   
    return moveCount > 0;
}

__device__ void simulateGameLeaf(curandState* deviceState, CudaSimulator& simulator, CudaGameState& state, size_t* wins, size_t* visits)
{
    Player startingPlayer = state.currentPlayer;
    size_t passCounter = 0;
    size_t limit = 0;
    while (limit < 128)
    {
        bool result = !doStep(state, simulator);
        if (result)
        {
            passCounter++;
            //__syncthreads();
            // if (threadIdx.x == 0)
            //     printf("Block %d: Raise counter to %lu on %lu\n", blockIdx.x, passCounter, limit);
            if (passCounter > 1)
                break;
        }
        else
        {
            passCounter = 0;
            // __syncthreads();
            // if (threadIdx.x == 0)
            //     printf("Block %d: Reset counter to %lu on %lu\n", blockIdx.x, passCounter, limit);
        }
        limit++;
    }
    if (threadIdx.x == 0)
    if (passCounter < 2)
        printf("Block %d unexpected exited game\n", blockIdx.x);
    else
        printf("Block %d exited game\n", blockIdx.x);
    __syncthreads();

    ++(*visits);
    if (state.isWinner(startingPlayer))
    {
        ++(*wins);
    }
    
}

__global__ void simulateGameLeaf(curandState* deviceState, Field* playfield, Player currentPlayer, size_t* wins, size_t* visits)
{
    int playfieldIndex = threadIdx.x;

    __shared__ Field sharedPlayfield[FIELD_DIMENSION * FIELD_DIMENSION];
    __shared__ bool possibleMoves[FIELD_DIMENSION*FIELD_DIMENSION];
    sharedPlayfield[playfieldIndex] = playfield[playfieldIndex];

    CudaGameState state =  { 
        sharedPlayfield, 
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
    __shared__ size_t node;
    int threadGroup = blockIdx.x;
    int playfieldIndex = threadIdx.x;
    if (threadIdx.x == 0) 
    {
        node = randomNumber(deviceStates, numberOfPlayfields);
    }

    __shared__ Field sharedPlayfield[FIELD_DIMENSION * FIELD_DIMENSION];
    __shared__ bool possibleMoves[FIELD_DIMENSION*FIELD_DIMENSION];
    
    size_t playfieldOffset = FIELD_DIMENSION * FIELD_DIMENSION * threadGroup;
    sharedPlayfield[playfieldIndex] = playfields[playfieldOffset + playfieldIndex];

    CudaGameState state =  { 
        sharedPlayfield, 
        possibleMoves, 
        FIELD_DIMENSION * FIELD_DIMENSION, 
        FIELD_DIMENSION, 
        currentPlayer 
    };
    CudaSimulator simulator(&state, deviceStates);
    OthelloResult& result = results[node];
    //OthelloResult result;
    size_t wins = 0;
    size_t visits = 0;
        if (threadIdx.x == 0)
        printf("Block %d started game on %lu\n", blockIdx.x, node);
    __syncthreads();
    simulateGameLeaf(deviceStates, simulator, state, &wins, &visits);
    __syncthreads();
        
    __syncthreads();
    if (threadIdx.x == 0)
    {
        results[node].wins += wins;
        results[node].visits += visits;
        //printf("TEST %d\n", result.wins);
        printf("Block %d finished game on %lu with %lu wins in %lu visits \n", blockIdx.x, node, results[node].wins, results[node].visits);
    }
}

__global__ void testDoStep(curandState* deviceState, Field* playfield, Player currentPlayer, float fakedRandom)
{
    int playfieldIndex = threadIdx.x;
    __shared__ Field sharedPlayfield[FIELD_DIMENSION * FIELD_DIMENSION];
    __shared__ bool possibleMoves[FIELD_DIMENSION*FIELD_DIMENSION];
    sharedPlayfield[playfieldIndex] = playfield[playfieldIndex];

    CudaGameState state =  { 
        sharedPlayfield, 
        possibleMoves, 
        FIELD_DIMENSION * FIELD_DIMENSION, 
        FIELD_DIMENSION, 
        currentPlayer 
    };
    CudaSimulator simulator(&state, deviceState);

    doStep(state, simulator, fakedRandom);

    playfield[playfieldIndex] = sharedPlayfield[playfieldIndex];
}

__global__ void testSimulateGameLeaf(curandState* deviceState, Field* playfield, Player currentPlayer, size_t* wins, size_t* visits)
{
    int playfieldIndex = threadIdx.x;

    __shared__ Field sharedPlayfield[FIELD_DIMENSION * FIELD_DIMENSION];
    __shared__ bool possibleMoves[FIELD_DIMENSION*FIELD_DIMENSION];
    sharedPlayfield[playfieldIndex] = playfield[playfieldIndex];

    CudaGameState state =  { 
        sharedPlayfield, 
        possibleMoves, 
        FIELD_DIMENSION * FIELD_DIMENSION, 
        FIELD_DIMENSION, 
        currentPlayer 
    };
    CudaSimulator simulator(&state, deviceState);
    simulateGameLeaf(deviceState, simulator, state, wins, visits);
    
	playfield[playfieldIndex] = sharedPlayfield[playfieldIndex];
}