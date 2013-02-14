#pragma once

#include <cassert>

#define THREAD_WATCHED (threadIdx.x == 0 || threadIdx.x == 32)

#define cassert(CONDITION, MESSAGE, ...) if (!(CONDITION)) printf(MESSAGE, __VA_ARGS__), assert(CONDITION)

const int FIELD_DIMENSION = 8;



__device__ size_t randomNumber(curandState* deviceStates, size_t maximum)
{
	curandState deviceState = deviceStates[blockIdx.x];
	size_t value = curand_uniform(&deviceState) * maximum;
	deviceStates[0] = deviceState;
    return value;
}   

