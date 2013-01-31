#include "MonteCarloTreeSearch.h"
#include <curand.h>
#include <curand_kernel.h>
#include "State.cuh"
#include "Move.cuh"
#include "OthelloField.h"

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
	size_t firstLevelStatesLength = untriedMoves.length;
	State* firstLevelStates = new State[firstLevelStatesLength];
	for (size_t i = 0; i < firstLevelStatesLength; ++i)
	{
		firstLevelStates[i] = State(rootState, untriedMoves.data[i]);
	}

	__syncthreads();

	size_t threadIterations = reiterations * 1.0f / NUMBER_OF_THREADS;
	for (size_t i = 0; i < threadIterations; ++i)
	{
		// select
		size_t stateIndex = randomNumber(deviceStates, firstLevelStatesLength);
		MoveVector possibleMoves = firstLevelStates[stateIndex].getPossibleMoves();
		Move move = possibleMoves.data[randomNumber(deviceStates, possibleMoves.length)];

		State secondLevelState = State(firstLevelStates[stateIndex], move);

		while (secondLevelState.isGameActive())
		{
			MoveVector furtherMoves = secondLevelState.getPossibleMoves();
			Move furtherMove = furtherMoves.data[randomNumber(deviceStates, furtherMoves.length)];
			secondLevelState.doMove(furtherMove);
		}
		// backpropagate
		firstLevelStates[stateIndex].updateFor(secondLevelState.getWinner());
	}
	// backpropagate II
	moveX[0] = 0;
	moveY[0] = 0;
	wins[0] = 0;
	visits[0] = 0;
	float best = 0;
	for (size_t i = 0; i < firstLevelStatesLength; ++i)
	{
		size_t stateWins = firstLevelStates[i].getWinsFor(rootState.currentPlayer());
		size_t stateVisits = firstLevelStates[i].getVisits();
		if (stateWins * 1.0f / stateVisits >= best)
		{
			best = stateWins * 1.0f / stateVisits;
			moveX[0] = firstLevelStates[i].getTriggerMoveX();
			moveY[0] = firstLevelStates[i].getTriggerMoveY();
			wins[0] = stateWins;
			visits[0] = stateVisits;
		}
	}
	delete[] firstLevelStates;
}

__global__ void computeKernel(curandState* deviceStates, size_t reiterations, size_t dimension, Field* playfield, size_t* moveX, size_t* moveY, size_t* wins, size_t* visits)
{
	State rootState(playfield, dimension, White);
	startGame(deviceStates, rootState, reiterations, moveX, moveY, wins, visits);
}