#pragma once

#include <cassert>

#define THREAD_WATCHED (threadIdx.x == 0 )

#define cassert(CONDITION, MESSAGE, ...) if (!(CONDITION)) printf(MESSAGE, __VA_ARGS__), assert(CONDITION)

const int FIELD_DIMENSION = 8;



__device__ size_t randomNumber(curandState* deviceStates, size_t maximum)
{
	size_t threadGeneratorIndex = threadIdx.x;
	curandState deviceState = deviceStates[threadGeneratorIndex];
	size_t value = curand_uniform(&deviceState) * maximum;
	deviceStates[threadGeneratorIndex] = deviceState;
    return value;
}   

