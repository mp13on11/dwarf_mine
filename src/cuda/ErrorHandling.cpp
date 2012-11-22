#include "ErrorHandling.h"

#include "CudaError.h"



namespace CudaUtils 
{
    void CheckError(cudaError_t state) 
    {
        if (state != cudaSuccess) 
        {
            throw CudaError(state);
        }
    }

    void CheckError(CUTBoolean result) 
    {
        if (result != CUTTrue) 
        {
            throw std::runtime_error("CUDA util error");
        }
    }

    void CheckState() 
    {
        cudaError_t state = cudaGetLastError();
        CheckError(state);
    }
}
