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
