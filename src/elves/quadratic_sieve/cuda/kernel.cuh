#pragma once

#include "Number.cuh"

__global__ void factorizeKernel(PNumData input, PNumData output);
