#include "ErrorHandling.h"

#include "CudaError.h"



namespace CudaUtils 
{
    void checkError(cudaError_t state) 
    {
        if (state != cudaSuccess) 
        {
            throw CudaError(state);
        }
    }

    void checkState() 
    {
        cudaError_t state = cudaGetLastError();
        checkError(state);
    }
}
