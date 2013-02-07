#include "MonteCarloTreeSearch.h"
#include <curand.h>
#include <curand_kernel.h>
#include "State.cuh"
#include "Move.cuh"
#include "OthelloField.h"
#include <stdio.h>

const int FIELD_DIMENSION = 8;

const bool FAKE_RANDOM = true;

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
	
    if(FAKE_RANDOM)
    {
        value = maximum / 2;
        printf("FAKE RANDOM TO %d \n", value);
    }
    return value;
}   

__device__ void startGame(curandState* deviceStates, State& rootState, size_t reiterations, size_t* moveX, size_t* moveY, size_t* wins, size_t* visits)
{
	// expand
	MoveVector untriedMoves = rootState.getPossibleMoves();
	
}

__global__ void computeKernel(curandState* deviceStates, size_t reiterations, size_t dimension, Field* playfield, size_t* moveX, size_t* moveY, size_t* wins, size_t* visits)
{
	printf("Thread %d\n", threadIdx.x);
	State rootState(playfield, dimension, White);
	startGame(deviceStates, rootState, reiterations, moveX, moveY, wins, visits);
}




__device__ Player getEnemyPlayer(Player currentPlayer)
{
	return (currentPlayer == Black ? White : Black);
}

__device__ void findPossibleMoves(bool* possibleMoves, Field* playfield, size_t playfieldIndex, int directionX, int directionY, Player currentPlayer)
{
    bool look = true;
    bool foundEnemy = false;
    Player enemyPlayer = getEnemyPlayer(currentPlayer);
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
        // printf("Random: ");
        // for (int i = 0; i < 64; i++)
        //     printf(" %d; ", possibleMoves[i]);
        // printf("\n");
	   randomIndex = getRandomMoveIndex(state, possibleMoves, moveCount);
    }
	__syncthreads();
	return randomIndex;
}

__device__ size_t sum(size_t* counts, size_t playfieldIndex, size_t arraySize)
{
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
	// __syncthreads();
 //    __shared__ size_t moves[FIELD_DIMENSION * FIELD_DIMENSION];
 //    moves[playfieldIndex] = possibleMoves[playfieldIndex] ? 1 : 0;
	// return sum(moves, playfieldIndex, FIELD_DIMENSION * FIELD_DIMENSION);
    
    __syncthreads();
    size_t sum = 0;
    for (int i = 0; i < FIELD_DIMENSION * FIELD_DIMENSION; i++)
    {
        if (possibleMoves[i])
        {
            sum++;
            __syncthreads();
        }
    }
    return sum;
}

__device__ void flipDirection(Field* playfield, size_t playfieldIndex, size_t moveIndex, int directionX, int directionY, Player currentPlayer)
{
    int currentIndex = playfieldIndex;
    Player enemyPlayer = getEnemyPlayer(currentPlayer);
    bool flip = false;

    for (currentIndex = playfieldIndex; currentIndex < FIELD_DIMENSION*FIELD_DIMENSION && currentIndex >= 0; currentIndex += directionY * FIELD_DIMENSION + directionX)
    {
    	if(playfield[currentIndex] != enemyPlayer)
    	{
    		flip = (playfield[currentIndex] == currentPlayer && currentIndex != playfieldIndex);
   			break;
    	}
    }
    __syncthreads();
    if (flip)
    {
	    for (; currentIndex - moveIndex != 0 ; currentIndex -= directionY * FIELD_DIMENSION + directionX)
	    {
	    	playfield[currentIndex] = currentPlayer;
	    }
	}

}

__device__ void flipEnemyCounter(Field* playfield, size_t playfieldIndex, size_t moveIndex, Player currentPlayer)
{
	int directionX = playfieldIndex % FIELD_DIMENSION - moveIndex % FIELD_DIMENSION;
    int directionY = playfieldIndex / FIELD_DIMENSION - moveIndex / FIELD_DIMENSION;

    if (abs(directionX) <= 1 && abs(directionY) <= 1 && moveIndex != playfieldIndex)
    {
    	flipDirection(playfield, playfieldIndex, moveIndex, directionX, directionY, currentPlayer);
    }
}

__global__ void computeSingleMove(curandState* deviceState, Field* playfield, size_t moveX, size_t moveY, int directionX, int directionY, Player currentPlayer)
{
    int playfieldIndex = threadIdx.x;
    int debug = 0;

    __shared__ Field sharedPlayfield[FIELD_DIMENSION * FIELD_DIMENSION];
    __shared__ bool possibleMoves[FIELD_DIMENSION*FIELD_DIMENSION];

    sharedPlayfield[playfieldIndex] = playfield[playfieldIndex];

    __syncthreads();

    calculatePossibleMoves(possibleMoves, sharedPlayfield, playfieldIndex, currentPlayer);
    size_t moveCount = countPossibleMoves(possibleMoves, playfieldIndex, sharedPlayfield);
    size_t limit = 100;

    while (moveCount > 0 && debug < limit + 1)
    {
        __syncthreads();

        size_t index = getRandomMoveIndex(deviceState, possibleMoves, moveCount, playfieldIndex);

        __syncthreads();        
        
        flipEnemyCounter(sharedPlayfield, playfieldIndex, index, currentPlayer);

        __syncthreads();

        sharedPlayfield[index] = currentPlayer;
        currentPlayer = getEnemyPlayer(currentPlayer);

        calculatePossibleMoves(possibleMoves, sharedPlayfield, playfieldIndex, currentPlayer);
        moveCount = countPossibleMoves(possibleMoves, playfieldIndex, sharedPlayfield);
        debug++;
	};

    if (playfieldIndex == 0 && moveCount== 0)
        printf("Runs: %d\n", debug);
    
    __syncthreads();
    
	playfield[playfieldIndex] = sharedPlayfield[playfieldIndex];
}