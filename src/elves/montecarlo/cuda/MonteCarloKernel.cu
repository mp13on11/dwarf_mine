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

__global__ void setupStateForRandom(curandState* state, unsigned long seed)
{
	int id =  threadIdx.x;
	curand_init(seed, id, 0, &state[id]);
}

__device__ bool doStep(CudaGameState& state, CudaSimulator& simulator, float fakedRandom = -1)
{
    __syncthreads();

    simulator.calculatePossibleMoves();
    size_t moveCount = simulator.countPossibleMoves();
    // if (threadIdx.x == 0)
    //     printf("Block %d: Moves %lu \n", blockIdx.x, moveCount);
    if (moveCount > 0)
    {
        __shared__ size_t index;
        if (threadIdx.x == 0)
            index = simulator.getRandomMoveIndex(moveCount, fakedRandom);

        state.oldField[threadIdx.x] = state.field[threadIdx.x];
        
        __syncthreads();

        simulator.flipEnemyCounter(index);

        __syncthreads();
        state.field[index] = state.currentPlayer;
        
    }
    state.currentPlayer = state.getEnemyPlayer();
   
    return moveCount > 0;
}

__device__ bool doStep(CudaGameState& state, CudaSimulator& simulator, size_t limit, float fakedRandom = -1)
{
    __syncthreads();

    simulator.calculatePossibleMoves();
    size_t moveCount = simulator.countPossibleMoves();
    if (moveCount > 0)
    {
        __shared__ size_t index;
        if (threadIdx.x == 0)
        {
            index = simulator.getRandomMoveIndex(moveCount, fakedRandom);
            // printf("Block %d: %lu move %lu (%lu,%lu) of %lu\n", blockIdx.x, limit, index, index % FIELD_DIMENSION, index / FIELD_DIMENSION, moveCount);
            
            // for (size_t i = 0; i < state.size; i++)
            // {
            //     if (state.field[i] == White)
            //     {
            //         printf("Block %d: %lu currently %d white \n", blockIdx.x, limit, i);
            //     }
            //     if (state.field[i] == Black)
            //     {
            //         printf("Block %d: %lu currently %d black \n", blockIdx.x, limit,i);
            //     }
            // }
            // for (size_t i = 0; i < state.size; i++)
            // {
            //     if (state.possible[i])
            //         printf("Block %d: %lu possible move %lu\n", blockIdx.x, limit, i);
            // }
        }

       

        state.oldField[threadIdx.x] = state.field[threadIdx.x];
        
        __syncthreads();

        simulator.flipEnemyCounter(index);

        __syncthreads();
        state.field[index] = state.currentPlayer;
        __syncthreads();
        
         // if (threadIdx.x == 0)
         // {
         //    bool same = true;
         //    for (int i = 0; i < state.size; i++)
         //    {
         //        same &= state.oldField[i] == state.field[i];
         //    }
         //    if (same)
         //    {
         //        printf("Block %d: %lu detected unchanged state\n", blockIdx.x, limit);
         //    }
         // }
        
    }
    else
    {
        // if (threadIdx.x == 0)
        // {
        //     printf("Block %d: %lu no move\n", blockIdx.x, limit);
        // }

    }
    state.currentPlayer = state.getEnemyPlayer();
   
    return moveCount > 0;
}

__device__ void simulateGameLeaf(curandState* deviceState, CudaSimulator& simulator, CudaGameState& state, size_t* wins, size_t* visits)
{
    Player startingPlayer = state.currentPlayer;
    size_t passCounter = 0;
    size_t rounds = 0;
    while (passCounter < 2)
    {
        bool passedMove = !doStep(state, simulator);
        passCounter = (passedMove ? passCounter + 1 : 0);
        
        assert(rounds < 128); // an impossible condition - it would mean that for every field both players had to pass
    }
    __syncthreads();
    // if (threadIdx.x == 0)
    //     if (passCounter < 2)
    //         printf("Block %d unexpected exited game %lu\n", blockIdx.x, limit);
    //     else
    //         printf("Block %d exited game %lu\n", blockIdx.x, limit);
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
    __shared__ size_t node;
    int threadGroup = blockIdx.x;
    int playfieldIndex = threadIdx.x;
    if (threadIdx.x == 0) 
    {
        node = randomNumber(deviceStates, numberOfPlayfields);
    }

    __shared__ Field sharedPlayfield[FIELD_DIMENSION * FIELD_DIMENSION];
    __shared__ Field oldPlayfield[FIELD_DIMENSION * FIELD_DIMENSION];
    __shared__ bool possibleMoves[FIELD_DIMENSION*FIELD_DIMENSION];
    
    size_t playfieldOffset = FIELD_DIMENSION * FIELD_DIMENSION * threadGroup;
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
    // if (threadIdx.x == 0)
    //     printf("Block %d started game on %lu\n", blockIdx.x, node);
    __syncthreads();
    simulateGameLeaf(deviceStates, simulator, state, &wins, &visits);
    __syncthreads();
        
    __syncthreads();
    if (threadIdx.x == 0)
    {
        results[node].wins += wins;
        results[node].visits += visits;
        //printf("TEST %d\n", result.wins);
        // printf("Block %d finished game on %lu with %lu wins in %lu visits \n", blockIdx.x, node, results[node].wins, results[node].visits);
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

    doStep(state, simulator, fakedRandom);

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