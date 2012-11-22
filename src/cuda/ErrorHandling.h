#pragma once

#include <cuda.h>
#include <cutil_inline.h>
#include <cuda_runtime_api.h>

#define CUDA_CHECK(_call)   \
    _call;                  \
    CheckState();

namespace CudaUtils {
    /************************************************************************/
    /* Check Cuda or CUTIL return code, throws CudaError
    /* if code indicates an error
    /************************************************************************/
    void CheckError(cudaError_t state);
    void CheckError(CUTBoolean result);

    /************************************************************************/
    /* Checks cudaGetLastError() and throws CudaError if error detected
    /************************************************************************/
    void CheckState();
}
