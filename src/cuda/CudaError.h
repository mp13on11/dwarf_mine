#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdexcept>

namespace CudaUtils 
{

    /************************************************************************/
    /* Exception class for Cuda error codes                                                                      
    /************************************************************************/
    class CudaError : public std::runtime_error
    {
    public:
        explicit CudaError(cudaError_t errorCode) : 
            m_errorCode(errorCode), std::runtime_error(cudaGetErrorString(errorCode))
        {}

    private:
        cudaError_t m_errorCode;
    };

}
