#pragma once

const int FIELD_DIMENSION = 8;

__device__ size_t randomNumber(curandState* deviceStates, size_t maximum)
{
	curandState deviceState = deviceStates[blockIdx.x];
	size_t value = curand_uniform(&deviceState) * maximum;
	deviceStates[0] = deviceState;
    return value;
}   

