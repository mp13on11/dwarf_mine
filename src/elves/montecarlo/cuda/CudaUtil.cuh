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




__device__ size_t randomNumber(curandState* deviceStates, size_t maximum)
{
	size_t threadGeneratorIndex = blockIdx.x;
	curandState deviceState = deviceStates[threadGeneratorIndex];
	size_t value = curand_uniform(&deviceState) * maximum;
	deviceStates[threadGeneratorIndex] = deviceState;
    return value;
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