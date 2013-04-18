#pragma once

#include "ErrorHandling.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstddef>


namespace CudaUtils
{

    /************************************************************************
     * Abstraction for Cuda memory management
     *
     * Manages allocation and transfer of memory
     ************************************************************************/
    template<typename MemType>
    class Memory
    {
    public:
        Memory() :
            gpuPtr(nullptr),
            size(0)
        {

        }

        explicit Memory(std::size_t size) :
            gpuPtr(allocate(size)),
            size(size)
        {
        }

        ~Memory()
        {
            free();
        }

        Memory(const Memory&) = delete;
        Memory& operator=(const Memory&) = delete;
        Memory(Memory&& other) :
            gpuPtr(other.gpuPtr),
            size(other.size)
        {
            other.gpuPtr = nullptr;
            other.size = 0;
        }

        void reallocate(std::size_t newSize)
        {
            auto newPtr = allocate(newSize);
            free();
            gpuPtr = newPtr;
            size = newSize;
        }

        void transferTo(MemType* hostPtr) const
        {
            checkError(
                cudaMemcpy(hostPtr, gpuPtr, sizeInBytes(), cudaMemcpyDeviceToHost)
            );
        }

        void transferFrom(const MemType* hostPtr)
        {
            checkError(
                cudaMemcpy(gpuPtr, hostPtr, sizeInBytes(), cudaMemcpyHostToDevice)
            );
        }

        const MemType* get() const
        {
            return gpuPtr;
        }

        MemType* get()
        {
            return gpuPtr;
        }
        
        std::size_t numberOfElements() const
        {
            return size;
        }

    private:
        std::size_t sizeInBytes() const
        {
            return size * sizeof(MemType);
        }

        MemType* allocate(std::size_t size)
        {
            MemType* ptr;
            checkError(cudaMalloc(&ptr, size * sizeof(MemType)));
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
