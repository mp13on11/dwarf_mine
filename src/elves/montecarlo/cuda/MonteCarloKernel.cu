#include "MonteCarloTreeSearch.h"
#include <curand.h>
#include <curand_kernel.h>
#include "State.cuh"
#include "Move.cuh"
#include "OthelloField.h"
#include <stdio.h>
//TODO use CORRECT values
const int NUMBER_OF_THREADS = 1;
__global__ void setupKernel(curandState* state, unsigned long seed)
{
	int id = 0; // threadIdx.x;
	curand_init(seed, id, 0, &state[id]);
}

__device__ size_t randomNumber(curandState* deviceStates, size_t maximum)
{
	curandState deviceState = deviceStates[0];
	size_t value = curand_uniform(&deviceState) * maximum;
	deviceStates[0] = deviceState;
	return value;
}

__device__ void startGame(curandState* deviceStates, State& rootState, size_t reiterations, size_t* moveX, size_t* moveY, size_t* wins, size_t* visits)
{
	// expand
	MoveVector untriedMoves = rootState.getPossibleMoves();
	// size_t firstLevelStatesLength = untriedMoves.length;
	// State* firstLevelStates = new State[firstLevelStatesLength];
	// for (size_t i = 0; i < firstLevelStatesLength; ++i)
	// {
	// 	firstLevelStates[i] = State(rootState, untriedMoves.data[i]);
	// }

	// __syncthreads();

	// size_t threadIterations = reiterations * 1.0f / NUMBER_OF_THREADS;
	// for (size_t i = 0; i < threadIterations; ++i)
	// {
	// 	// select
	// 	size_t stateIndex = randomNumber(deviceStates, firstLevelStatesLength);
	// 	MoveVector possibleMoves = firstLevelStates[stateIndex].getPossibleMoves();
	// 	Move move = possibleMoves.data[randomNumber(deviceStates, possibleMoves.length)];

	// 	State secondLevelState = State(firstLevelStates[stateIndex], move);

	// 	while (secondLevelState.isGameActive())
	// 	{
	// 		MoveVector furtherMoves = secondLevelState.getPossibleMoves();
	// 		Move furtherMove = furtherMoves.data[randomNumber(deviceStates, furtherMoves.length)];
	// 		secondLevelState.doMove(furtherMove);
	// 	}
	// 	// backpropagate
	// 	firstLevelStates[stateIndex].updateFor(secondLevelState.getWinner());
	// }
	// // backpropagate II
	// moveX[0] = 0;
	// moveY[0] = 0;
	// wins[0] = 0;
	// visits[0] = 0;
	// float best = 0;
	// for (size_t i = 0; i < firstLevelStatesLength; ++i)
	// {
	// 	size_t stateWins = firstLevelStates[i].getWinsFor(rootState.currentPlayer());
	// 	size_t stateVisits = firstLevelStates[i].getVisits();
	// 	if (stateWins * 1.0f / stateVisits >= best)
	// 	{
	// 		best = stateWins * 1.0f / stateVisits;
	// 		moveX[0] = firstLevelStates[i].getTriggerMoveX();
	// 		moveY[0] = firstLevelStates[i].getTriggerMoveY();
	// 		wins[0] = stateWins;
	// 		visits[0] = stateVisits;
	// 	}
	// }
	// delete[] firstLevelStates;
}

__global__ void computeKernel(curandState* deviceStates, size_t reiterations, size_t dimension, Field* playfield, size_t* moveX, size_t* moveY, size_t* wins, size_t* visits)
{
	printf("Thread %d\n", threadIdx.x);
	State rootState(playfield, dimension, White);
	startGame(deviceStates, rootState, reiterations, moveX, moveY, wins, visits);
}

const int FIELD_DIMENSION = 8;

// __global__ bool flip(int playfieldX, int playfieldY, Field* playfield, size_t moveX, size_t moveY, int directionX, int directionY, int offset, Field expectedPlayer)
// {
//     int oldMovedX = moveX + directionX * offset - 1;
//     int oldMovedY = moveY + directionY * offset - 1;
//     int oldMovedIndex = moveX + moveY * FIELD_DIMENSION;
//     int movedX = oldMovedX + directionX;
//     int movedY = oldMovedY + directionY;
//     int movedIndex = moveX + directionX + (moveY + directionY) * FIELD_DIMENSION;
//     bool allowed = true;
//     if (playfieldX == movedX && playfieldY == movedY)
//     {
//         if (playfield[movedIndex] == 0) && playfield[oldMovedIndex] != playfield[movedIndex])
//         {
            
//         }
//     }
// return true;
// }

__device__ void findPossibleMoves(bool* possibleMoves, Field* playfield, size_t playfieldX, size_t playfieldY, size_t playfieldIndex, int directionX, int directionY, Player currentPlayer)
{
    bool look = true;
    while (look)
    {
        int neighbourX = playfieldX + directionX;
        int neighbourY = playfieldY + directionY;
        int neighbourIndex = neighbourY * FIELD_DIMENSION + playfieldX;
        if (neighbourX < FIELD_DIMENSION && neighbourX >= 0 && neighbourY < FIELD_DIMENSION && neighbourY >= 0)
        {
            if (playfield[neighbourIndex] == Free)
            {
                possibleMoves[playfieldIndex] |= false;
                look = false;
            }
            else if (playfield[neighbourIndex] == currentPlayer)
            {
                possibleMoves[playfieldIndex] |= true;
                look = false;
            }
        }
        else
        {
            possibleMoves[playfieldIndex] |= false;
            look = false;
        }
        directionX++;
        directionY++;
}
}

__global__ void computeSingleMove(Field* playfield, size_t moveX, size_t moveY, int directionX, int directionY, Player currentPlayer)
{
    int playfieldIndex = threadIdx.x;
    int playfieldX = threadIdx.x % FIELD_DIMENSION;
    int playfieldY = threadIdx.x / FIELD_DIMENSION;

    Player enemyPlayer = (currentPlayer == Black ? White : Black);
    //Field sharedPlayfield[FIELD_DIMENSION * FIELD_DIMENSION];
    Field* sharedPlayfield = playfield;
    bool possibleMoves[FIELD_DIMENSION*FIELD_DIMENSION];
    // printf("Block %2d %2d, Thread %2d %2d\n", 
    //  blockIdx.x, blockIdx.y, playfieldX, playfieldY);
    //sharedPlayfield[playfieldIndex] = playfield[playfieldIndex];
    possibleMoves[playfieldIndex] = false;

    __syncthreads();

    if (sharedPlayfield[playfieldIndex] == Free)
    {
        findPossibleMoves(possibleMoves, sharedPlayfield, playfieldX, playfieldY, playfieldIndex,  1,  1, currentPlayer);
        findPossibleMoves(possibleMoves, sharedPlayfield, playfieldX, playfieldY, playfieldIndex,  1,  0, currentPlayer);
        findPossibleMoves(possibleMoves, sharedPlayfield, playfieldX, playfieldY, playfieldIndex,  1, -1, currentPlayer);
        findPossibleMoves(possibleMoves, sharedPlayfield, playfieldX, playfieldY, playfieldIndex,  0,  1, currentPlayer);
        
        findPossibleMoves(possibleMoves, sharedPlayfield, playfieldX, playfieldY, playfieldIndex,  0, -1, currentPlayer);
        findPossibleMoves(possibleMoves, sharedPlayfield, playfieldX, playfieldY, playfieldIndex, -1,  1, currentPlayer);
        findPossibleMoves(possibleMoves, sharedPlayfield, playfieldX, playfieldY, playfieldIndex, -1,  0, currentPlayer);
        findPossibleMoves(possibleMoves, sharedPlayfield, playfieldX, playfieldY, playfieldIndex, -1, -1, currentPlayer);
    }

    // printf("Block %2d %2d, Thread %2d %2d, %3d, Field: %d\n", 
    //     blockIdx.x, blockIdx.y, playfieldX, playfieldY, playfieldIndex, (int)sharedPlayfield[playfieldX][playfieldY] );
    
	__syncthreads();

	if (possibleMoves[playfieldIndex])
	{
		printf("Block %2d %2d, Thread %2d %2d %2d, Field: %d [%2u,%2u]\n", 
		  blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, threadIdx.z, 
		  sharedPlayfield[playfieldIndex] , playfieldX, playfieldY);
	}
}