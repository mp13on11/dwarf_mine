/*****************************************************************************
* Dwarf Mine - The 13-11 Benchmark
*
* Copyright (c) 2013 Bünger, Thomas; Kieschnick, Christian; Kusber,
* Michael; Lohse, Henning; Wuttke, Nikolai; Xylander, Oliver; Yao, Gary;
* Zimmermann, Florian
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*****************************************************************************/

#pragma once

#define FIELD_DIMENSION 8
#define FIELD_SIZE FIELD_DIMENSION * FIELD_DIMENSION


#include "Debug.cuh"

__device__ size_t randomNumber(float* randomValues, size_t* randomSeed, size_t limit, float fakedRandom = -1)
{
	float random = 0;
	if (fakedRandom >= 0)
	{
		random = fakedRandom;
	}
	else
	{
		random = randomValues[*randomSeed];
	}
	size_t value = size_t(floor(random * limit));
	if (value == limit)
	{
		--value;
	}	
	++(*randomSeed);
	return value;
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
	if (maximum == result) // nothing with (0,1] ... sometimes it is rounded to maximum - so we need to reduce manually
	{
		--result;
	}
	cassert(result < maximum, "Random %f - Maximum %lu = %f = %lu\n", random, maximum, random * maximum, result);
    return result;
} 

__device__ size_t numberOfMarkedFields(const bool* field)
{
	__shared__ int s[FIELD_DIMENSION];

	__syncthreads();
	if (threadIdx.x % FIELD_DIMENSION == 0) 
	{
		s[threadIdx.x / FIELD_DIMENSION] = 0;
	}
	
	__syncthreads();
	
	if (field[threadIdx.x])
	{
		atomicAdd(&s[threadIdx.x / FIELD_DIMENSION], 1u);
	}
	
	__syncthreads();
	
	if (threadIdx.x % FIELD_DIMENSION == 0 && threadIdx.x != 0)
	{
		atomicAdd(&s[0], s[threadIdx.x / FIELD_DIMENSION]);
	}

	__syncthreads();

	return (size_t)s[0];
}