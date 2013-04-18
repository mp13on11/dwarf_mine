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
#include "stdio.h"


__device__ Number pow(Number b, Number e)
{
    Number result = 1;
    for (int i = (NUM_FIELDS*32)-1; i>=0; --i)
    {
       result *= result;
       if (e.bitAt(i) == 1)
       {
           result *= b;
       }
    }
    return result;
}

__device__ uint32_t pow(Number b, Number e, Number m)
{
    Number result = 1;
    for (int i = (NUM_FIELDS*32)-1; i>=0; --i)
    {
       result *= result;
       result = result % m;
       if (e.bitAt(i) == 1)
       {
           result *= b;
           result = result % m;
       }
    }
    return result.get_ui();
}

__device__ uint32_t cuda_pow(uint32_t b, uint32_t e)
{
    uint32_t res = 1;
    while (e > 0)
    {
        while (e % 2 == 0)
        {
            e >>= 1;
            b *= b;
        }
        --e;
        res *= b;
    }
    return res;
}

__device__ float log(const Number& n)  
{
    uint32_t low = 0;
    uint32_t high = 0;
    uint32_t o;

    for (int i = NUM_FIELDS-1; i >= 0; --i)    
    {
        if (n.fields[i] & 0xFFFFFFFF)
        {
            if (i == 0) 
            {
                return log((float)n.fields[i]);
            }

            int l = 1;
            uint32_t copy = n.fields[i]; 
            while (copy >>= 1) 
            {
                l++;
            }
            high = (n.fields[i] << o);// & (0xFFFFFFFF << o);
            low = (n.fields[i-1] >> l);

            return log((float) (low | high)) + log(2.0f)*((i-1)*32+l);
        }
    }
    return 0;
}

__device__ float lb(Number x)
{
    return log(x)/log(2.0f);
}

__device__ uint32_t lb_scaled(Number x, uint32_t maxBits)
{
    uint32_t maxLogBits = (uint32_t)ceil(log((float)maxBits+1.0f)/log(2.0f));
    uint32_t scale_shift = 32 - maxLogBits;
    return (uint32_t)(lb(x) * (1 << scale_shift));
}

__device__ uint32_t log_2_22(Number x)
{
    return lb_scaled(x, 1023);
}

__global__ void megaKernel(const Number* number, uint32_t* logs, const uint32_t* factorBase, const int factorBaseSize, const Number* start, const Number* end, const uint32_t intervalLength)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    Number newStart(*start + index);
      
    if (index >= intervalLength) 
    {
        return;
    }
    for (int i=0; i<factorBaseSize; ++i)
    {
    	Number prime(factorBase[i]);
    	Number primePower(factorBase[i]);
    	
        int primeLog = log_2_22(prime) - 1;
        Number q = (newStart*newStart) - *number;
        
        if (primeLog > logs[index])
            break;

    	while (primePower < *number) 
        {
        	if ((q % primePower).isZero()) 
        	{        	   
                    logs[index] -= primeLog;
        	}
        	else
        	{
        	    break;
        	}
        	primePower *= prime;
        	
    	}
    } 
}

__global__ void testAddKernel(PNumData pLeft, PNumData pRight, PNumData output)
{
    Number left(pLeft);
    Number right(pRight);
    Number result(left + right);
    result.writeTo(output);
}


__global__ void testSubKernel(PNumData pLeft, PNumData pRight, PNumData output)
{
    Number left(pLeft);
    Number right(pRight);
    Number result(left - right);
    result.writeTo(output);
}

__global__ void testMulKernel(PNumData pLeft, PNumData pRight, PNumData output)
{
    Number left(pLeft);
    Number right(pRight);
    Number result(left * right);
    result.writeTo(output);
}

__global__ void testDivKernel(PNumData pLeft, PNumData pRight, PNumData output)
{
    Number left(pLeft);
    Number right(pRight);
    
    Number result(left / right);
    result.writeTo(output);
}

__global__ void testModKernel(PNumData pLeft, PNumData pRight, PNumData output)
{
    Number left(pLeft);
    Number right(pRight);
    Number result(left % right);
    result.writeTo(output);
}

__global__ void testSmallerThanKernel(PNumData pLeft, PNumData pRight, bool* output)
{
    Number left(pLeft);
    Number right(pRight);
    *output = left < right;
}

__global__ void testLargerThanKernel(PNumData pLeft, PNumData pRight, bool* output)
{
    Number left(pLeft);
    Number right(pRight);
    *output = left > right;
}

__global__ void testLargerEqualKernel(PNumData pLeft, PNumData pRight, bool* output)
{
    Number left(pLeft);
    Number right(pRight);
    *output = left >= right;
}

__global__ void testEqualKernel(PNumData pLeft, PNumData pRight, bool* output)
{
    Number left(pLeft);
    Number right(pRight);
    *output = left == right;
}

__global__ void testShiftLeftKernel(PNumData pLeft, uint32_t offset, PNumData output)
{
    Number left(pLeft);
    Number result(left << offset);
    result.writeTo(output);
}

__global__ void testShiftRightKernel(PNumData pLeft, uint32_t offset, PNumData output)
{
    Number left(pLeft);
    Number result(left >> offset);
    result.writeTo(output);
}

__global__ void testModPowKernel(PNumData pBase, PNumData pExponent, PNumData pMod, PNumData output)
{
    Number base(pBase);
    Number exponent(pExponent);
    Number mod(pMod);
    Number result(pow(base, exponent, mod));
    result.writeTo(output);
}

__global__ void testCudaPowKernel(int b, int e, int* output)
{
    int result(cuda_pow(b, e));
    *output = result;
}
