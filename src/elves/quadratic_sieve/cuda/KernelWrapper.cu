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

#include "kernel.cuh"
#include "Number.cuh"
#include "Factorize.h"
#include "common/Utils.h"
#include "stdio.h"

const size_t BLOCK_SIZE = 256;

void megaWrapper(const uint32_t* number, uint32_t* logs_d, const uint32_t* factorBase_d, const size_t factorBaseSize, const uint32_t* start, const uint32_t* end, const uint32_t intervalLength)
{
    size_t numThreads = intervalLength;
    size_t numBlocks = div_ceil(numThreads, BLOCK_SIZE);
    
    Number* number_d; 
    cudaMalloc(&number_d, sizeof(uint32_t)*NUM_FIELDS);
    cudaMemcpy(number_d, number, NUM_FIELDS*sizeof(uint32_t), cudaMemcpyHostToDevice);
       
	Number* start_d; 
	cudaMalloc(&start_d, sizeof(uint32_t)*NUM_FIELDS);
	cudaMemcpy(start_d, start, NUM_FIELDS*sizeof(uint32_t), cudaMemcpyHostToDevice);
	
	Number* end_d; 
	cudaMalloc(&end_d, sizeof(uint32_t)*NUM_FIELDS);
	cudaMemcpy(end_d, end, NUM_FIELDS*sizeof(uint32_t), cudaMemcpyHostToDevice);
	
	megaKernel<<<numBlocks, BLOCK_SIZE>>>(number_d, logs_d, factorBase_d, (int)factorBaseSize, start_d, end_d, intervalLength);
	cudaDeviceSynchronize();
	
	cudaFree(start_d);
	cudaFree(end_d);
	cudaFree(number_d);
}
