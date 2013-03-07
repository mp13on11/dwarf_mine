#pragma once

#include <cassert>

const int FIELD_DIMENSION = 8;

#define THREAD_WATCHED (threadIdx.x == 0  && blockIdx.x == 1)

#define cassert(CONDITION, MESSAGE, ...) \
	if (!(CONDITION)) \
	{	\
		printf(MESSAGE, __VA_ARGS__); \
		assert(CONDITION); \
	}

#define charfield(FIELD) (FIELD == Free ? ' ' : FIELD == Illegal ? '?' : FIELD == Black ? 'B' : 'W')

#define printplayfield(LIMIT, MESSAGE,PLAYFIELD) \
		for (size_t i = 0; i < FIELD_DIMENSION; ++i) \
		 	printf("%lu Block %d: %s Line %lu: \t%c %c %c %c %c %c %c %c\n", \
				LIMIT, blockIdx.x, MESSAGE, i, charfield(PLAYFIELD[i*FIELD_DIMENSION + 0]),  charfield(PLAYFIELD[i*FIELD_DIMENSION + 1]),  charfield(PLAYFIELD[i*FIELD_DIMENSION + 2]),  charfield(PLAYFIELD[i*FIELD_DIMENSION + 3]),  charfield(PLAYFIELD[i*FIELD_DIMENSION + 4]),  charfield(PLAYFIELD[i*FIELD_DIMENSION + 5]),  charfield(PLAYFIELD[i*FIELD_DIMENSION + 6]),  charfield(PLAYFIELD[i*FIELD_DIMENSION + 7]) \
			);
		

#define cstateassert(CONDITION, STATE, MESSAGE, LIMIT) \
	__syncthreads(); \
	if (!CONDITION && threadIdx.x == 0) \
	{ \
		printf(MESSAGE, blockIdx.x, LIMIT); \
		printf("\n");\
		printplayfield(LIMIT, "OLD", state.oldField); \
		printplayfield(LIMIT, "NEW", state.field); \
		assert(CONDITION); \
	}



/***
 * Delivers a random number x with 0 <= x < maximum
 */
__device__ size_t randomNumber(curandState* deviceStates, size_t maximum, float fakedRandom = -1)
{
	size_t threadGeneratorIndex = blockIdx.x;
	float random = 0;
	if (fakedRandom >= 0)
	{
		random = fakedRandom;
	}
	else
	{
		curandState deviceState = deviceStates[threadGeneratorIndex];
		random = 1.0f - curand_uniform(&deviceState); // delivers (0, 1] - we need [0, 1)
		deviceStates[threadGeneratorIndex] = deviceState;
	}
	size_t result = size_t(floor(random * maximum));
	//cassert(result < maximum, "Random %f - Maximum %lu = %f = %lu\n", random, maximum, random * maximum, result);
    return result;
}   


__device__ bool unchangedState(CudaGameState& state, size_t limit)
{
    bool same = true;
    for (int i = 0; i < state.size; i++)
    {
        same &= (state.oldField[i] == state.field[i]);
    }
    return same;
}