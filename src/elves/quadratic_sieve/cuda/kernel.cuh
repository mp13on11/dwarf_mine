#pragma once

#include "Number.cuh"

#define NUMBERS_PER_THREAD 32

__global__ void megaKernel(uint32_t* logs, const uint32_t* factorBase, const Number* start, const uint32_t intervalLength);
