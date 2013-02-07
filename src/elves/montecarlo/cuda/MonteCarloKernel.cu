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

__device__ void findPossibleMoves(bool* possibleMoves, Field* playfield, size_t playfieldIndex, int directionX, int directionY, Player currentPlayer)
{
    bool look = true;
    bool foundEnemy = false;
    Player enemyPlayer = (currentPlayer == Black ? White : Black);
    int playfieldX = playfieldIndex % FIELD_DIMENSION;
    int playfieldY = playfieldIndex / FIELD_DIMENSION;
    int neighbourX = playfieldX + directionX;
    int neighbourY = playfieldY + directionY;
    while (look)
    {
        int neighbourIndex = neighbourY * FIELD_DIMENSION + neighbourX;
        if (neighbourX < FIELD_DIMENSION && neighbourX >= 0 && neighbourY < FIELD_DIMENSION && neighbourY >= 0)
        {
            if (playfield[neighbourIndex] == Free)
            {
                possibleMoves[playfieldIndex] |= false;
                look = false;
            }
            else if(playfield[neighbourIndex] == enemyPlayer){
            	foundEnemy = true;
            }
            else if (playfield[neighbourIndex] == currentPlayer)
            {
                possibleMoves[playfieldIndex] |= foundEnemy;
                look = false;
            }
        }
        else
        {
            possibleMoves[playfieldIndex] |= false;
            look = false;
        }
        neighbourX += directionX;
        neighbourY += directionY;
	}
}

__device__  void calculatePossibleMoves(bool* possibleMoves, Field* sharedPlayfield, size_t playfieldIndex, Player currentPlayer)
{
    possibleMoves[playfieldIndex] = false;

    __syncthreads();

    if (sharedPlayfield[playfieldIndex] == Free)
    {
        findPossibleMoves(possibleMoves, sharedPlayfield, playfieldIndex,  1,  1, currentPlayer);
        findPossibleMoves(possibleMoves, sharedPlayfield, playfieldIndex,  1,  0, currentPlayer);
        findPossibleMoves(possibleMoves, sharedPlayfield, playfieldIndex,  1, -1, currentPlayer);
        findPossibleMoves(possibleMoves, sharedPlayfield, playfieldIndex,  0,  1, currentPlayer);
        
        findPossibleMoves(possibleMoves, sharedPlayfield, playfieldIndex,  0, -1, currentPlayer);
        findPossibleMoves(possibleMoves, sharedPlayfield, playfieldIndex, -1,  1, currentPlayer);
        findPossibleMoves(possibleMoves, sharedPlayfield, playfieldIndex, -1,  0, currentPlayer);
        findPossibleMoves(possibleMoves, sharedPlayfield, playfieldIndex, -1, -1, currentPlayer);
    }

    __syncthreads();
}



__device__ size_t getRandomMoveIndex(curandState* state, bool* possibleMoves, size_t moveCount)
{
	size_t randomMoveIndex = 0;
	if (moveCount > 1)
	{
		randomMoveIndex = randomNumber(state, moveCount);
	}
	size_t possibleMoveIndex = 0;
	for (size_t i = 0; i < FIELD_DIMENSION * FIELD_DIMENSION; ++i)
	{
		if (possibleMoves[i])
		{
			if (possibleMoveIndex == randomMoveIndex)
			{
				return i;
			}
			possibleMoveIndex++;;
		}
	}
	return 0;
}

__device__ size_t getRandomMoveIndex(curandState* state, bool* possibleMoves, size_t moveCount, size_t playfieldIndex)
{
	__shared__ size_t randomIndex;
	if (playfieldIndex == 0)
	{
		randomIndex = getRandomMoveIndex(state, possibleMoves, moveCount);
	}
	__syncthreads();
	return randomIndex;
}

__device__ size_t sum(size_t* counts, size_t playfieldIndex, size_t arraySize)
{
	__syncthreads();
	if (arraySize > 1)
	{
		arraySize = arraySize / 2;
		if (playfieldIndex < arraySize)
		{
			counts[playfieldIndex] += counts[playfieldIndex + arraySize];
		}
		return sum(counts, playfieldIndex, arraySize);
	}
	return counts[0];
}

__device__ size_t countPossibleMoves(bool* possibleMoves, size_t playfieldIndex, Field* playfield)
{
	// Serialize via CUDA. Possible optimization PARALLEL REDUCTION
	 __shared__ size_t moves[FIELD_DIMENSION * FIELD_DIMENSION];
	// __syncthreads();
	if(possibleMoves[playfieldIndex])
	{
		moves[playfieldIndex] = 1;
		if (moves[playfieldIndex] == 1){

			playfield[playfieldIndex] = Illegal;
		}
		//__syncthreads();
		printf("%d T__EST %d\n", playfieldIndex, 1);
	}
	__syncthreads();
	size_t s = sum(moves, playfieldIndex, FIELD_DIMENSION * FIELD_DIMENSION);
	// return moves[0];
	playfield[s % 8] = Illegal;
	size_t sum = 0;
	// __syncthreads();
	for (size_t i = 0; i < FIELD_DIMENSION * FIELD_DIMENSION; ++i)
	{
		if (possibleMoves[i])
		{
			++sum;
		}
	}
	return sum;
}

__global__ void computeSingleMove(curandState* deviceState, Field* playfield, size_t moveX, size_t moveY, int directionX, int directionY, Player currentPlayer)
{
    int playfieldIndex = threadIdx.x;
    //printf("Hello world from: %u\n", (size_t) playfieldIndex);
    //Field sharedPlayfield[FIELD_DIMENSION * FIELD_DIMENSION];
    Field* sharedPlayfield = playfield;
    __shared__ bool possibleMoves[FIELD_DIMENSION*FIELD_DIMENSION];
    // printf("Block %2d %2d, Thread %2d %2d\n", 
    //  blockIdx.x, blockIdx.y, playfieldX, playfieldY);
    //sharedPlayfield[playfieldIndex] = playfield[playfieldIndex];

    // printf("Block %2d %2d, Thread %2d %2d, %3d, Field: %d\n", 
    //     blockIdx.x, blockIdx.y, playfieldX, playfieldY, playfieldIndex, (int)sharedPlayfield[playfieldX][playfieldY] );
    
    calculatePossibleMoves(possibleMoves, playfield, playfieldIndex, currentPlayer);
    
    size_t moveCount = countPossibleMoves(possibleMoves, playfieldIndex, playfield);
    size_t index = FIELD_DIMENSION * FIELD_DIMENSION;
    if (moveCount > 0)
    {
    	index = getRandomMoveIndex(deviceState, possibleMoves, moveCount, playfieldIndex);
    }

	if (possibleMoves[playfieldIndex])
	{
		//sharedPlayfield[playfieldIndex] = Illegal;
		int playfieldX = playfieldIndex % FIELD_DIMENSION;
    	int playfieldY = playfieldIndex / FIELD_DIMENSION;
		printf("Block %2d %2d, Thread %2d %2d %2d, Field: %d [%2u,%2u] - Move %d\n", 
		  blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, threadIdx.z, 
		  sharedPlayfield[playfieldIndex] , playfieldX, playfieldY, index);
	}
}