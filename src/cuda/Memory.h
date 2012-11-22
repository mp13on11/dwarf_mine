#pragma once

#include "ErrorHandling.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstddef>


namespace CudaUtils
{

    /************************************************************************/
    /* Abstraction for Cuda memory management
    /*
    /* Manages allocation and transfer of memory
    /************************************************************************/
    template<typename MemType>
    class Memory
    {
    public:
        explicit Memory(std::size_t size) : gpuPtr(allocate(size)), size(size)
        {
        }

        ~Memory()
        {
            free();
        }
        
        Memory(const Memory&) = delete;
        Memory& operator=(const Memory&) = delete;    
        Memory(Memory&& other) : gpuPtr(other.gpuPtr), size(other.size)
        {
            other.gpuPtr = nullptr;
            other.size = 0;            
        }

        const MemType* get() const
        {
            return gpuPtr;
        }

        MemType* get()
        {
            return gpuPtr;
        }
        
    private:
        MemType* allocate(std::size_t size)
        {
            MemType* ptr;
            checkError(cudaMalloc(&ptr, size));
            return ptr;
        }
        
        void free()
        {
            if (gpuPtr != nullptr)
                cudaFree(gpuPtr);
        }

        MemType* gpuPtr;
        std::size_t size;
    };
    
}
