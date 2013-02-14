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

// __device__ bool doStep(CudaGameState& state, CudaSimulator& simulator, float fakedRandom = -1)
// {
//     __syncthreads();

//     simulator.calculatePossibleMoves();
//     size_t moveCount = simulator.countPossibleMoves();
    
//     if (moveCount > 0)
//     {
//         __shared__ size_t index;
//         if (threadIdx.x == 0)
//             index = simulator.getRandomMoveIndex(moveCount, fakedRandom);

//         state.oldField[threadIdx.x] = state.field[threadIdx.x];
        
//         __syncthreads();

//         simulator.flipEnemyCounter(index);

//         __syncthreads();
//         state.field[index] = state.currentPlayer;
        
//     }
//     state.currentPlayer = state.getEnemyPlayer();
   
//     return moveCount > 0;
// }


__device__ bool doStep(CudaGameState& state, CudaSimulator& simulator, size_t limit, float fakedRandom = -1)
{
    __syncthreads();

    simulator.calculatePossibleMoves();
    __syncthreads();
    size_t moveCount = simulator.countPossibleMoves();
    if (moveCount > 0)
    {
        printf("%d %lu Looking for Index \n",threadIdx.x,limit);
        __shared__ size_t index;
        index = 72;
        __syncthreads();
        // if (threadIdx.x == 0)
        // {
            index = simulator.getRandomMoveIndex(moveCount, fakedRandom);
        // }
            __syncthreads();
            printf("%d %lu Received Random Index %lu\n",threadIdx.x, limit, index);
        // else
        // {
        //     size_t trash = simulator.getRandomMoveIndex(moveCount, fakedRandom);
        //     printf("%d Waiting Random Index %lu \n",threadIdx.x, trash);   
        // }

        __syncthreads();
        __threadfence();
        printf("%d %lu Synched Index %lu\n",threadIdx.x, limit, index);
       cassert(index < state.size, "Detected unexpected move index %d for maximal index %lu in %d\n", index, state.size - 1, threadIdx.x);

        state.oldField[threadIdx.x] = state.field[threadIdx.x];
        __syncthreads();
        
        if (THREAD_WATCHED)
        {
            printf("Block %d [%d]: %lu move %lu (%lu,%lu) of %lu\n", blockIdx.x, threadIdx.x, limit, index, index % FIELD_DIMENSION, index / FIELD_DIMENSION, moveCount);
            
            for (size_t i = 0; i < state.size; i++)
            {
                if (state.field[i] == White)
                {
                    printf("Block %d [%d]: %lu currently %d white \n", blockIdx.x,threadIdx.x,  limit, i);
                }
                if (state.field[i] == Black)
                {
                    printf("Block %d [%d]: %lu currently %d black \n", blockIdx.x,threadIdx.x,  limit, i);
                }
            }
            for (size_t i = 0; i < state.size; i++)
            {
                if (state.possible[i])
                    printf("Block %d [%d]: %lu possible move %lu\n", blockIdx.x,threadIdx.x,  limit, i);
            }
        }

        simulator.flipEnemyCounter(index);

        __syncthreads();
        state.field[index] = state.currentPlayer;
        __syncthreads();
        
         if (THREAD_WATCHED)
         {
            bool same = true;
            for (int i = 0; i < state.size; i++)
            {
                same &= state.oldField[i] == state.field[i];
            }
            cassert(!same, "Block %d: %lu detected unchanged state\n", blockIdx.x, limit);
         }
        
    }
    // else
    // {
    //     if (THREAD_WATCHED)
    //     {
    //         printf("Block %d [%d]: %lu no move\n", blockIdx.x, threadIdx.x, limit);
    //     }

    // }
    state.currentPlayer = state.getEnemyPlayer();
    if (THREAD_WATCHED)
        printf("Block %d [%d]: Moves %lu \n", blockIdx.x, threadIdx.x, moveCount);
    return moveCount > 0;
}

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
        if (THREAD_WATCHED)
            printf("%2d PassCounter: %lu in %lu with move %s (%s)\n\n", threadIdx.x, passCounter, rounds, passedMove ? "TRUE" : "FALSE", state.currentPlayer == Black ? "Black" : "White");

        cassert (rounds++ < 1280, "Detected rounds overflow in %d\n", threadIdx.x); // an impossible condition - it would mean that for every field both players had to pass
        if (rounds > 1280) assert(false);
    }
    __syncthreads();
    if (THREAD_WATCHED)
        if (passCounter < 2)
            printf("Block %d unexpected exited game %lu\n", blockIdx.x, rounds);
        else
            printf("Block %d exited game %lu\n", blockIdx.x, rounds);
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