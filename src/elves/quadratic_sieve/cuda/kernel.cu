#include "kernel.cuh"
#include "Number.cuh"
#include "stdio.h"

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
    Number newStart(*start + index * NUMBERS_PER_THREAD);
    Number newStartP(newStart);
    //Number newStartN(newStart);
    Number newEnd = newStart + Number(NUMBERS_PER_THREAD);
      
    if (newStart > *end) 
    {
        return;
    }

    for (int i=0; i<factorBaseSize; ++i)
    {
    	Number prime(factorBase[i]);
    	Number primePower(prime);
    	
        int primeLog = log_2_22(prime) - 1;
    	while (primePower < *number) 
        {
            newStart = newStartP;
    	   
    	    int timesAdvanced = 0;
        	bool invalid = false; 	
        	
            Number q = (newStart*newStart) - *number;
        	while (!(q % primePower).isZero()) 
        	{        	   
        	   ++timesAdvanced;
        	   newStart += 1;
        	   if ((timesAdvanced > NUMBERS_PER_THREAD) || (newStart > *end)) 
        	   {
        	       invalid = true;
        	       break;
        	   }
               q = (newStart * newStart) - *number;
        	} 
            //newStartN = newStart + Number(1);
        	
        	if (invalid) break;
            //printf("q: %d, x: %d, n: %d\n", q.get_ui(), newStart.get_ui(), number->get_ui());
        	
        	Number offset = newStart - *start;
            if (index == 0)
            {
                printf("offset for prime %d %d\n", offset.get_ui(), primePower.get_ui());
            }
            int endOffset = offset.get_ui() + NUMBERS_PER_THREAD;
        	for (; offset.get_ui() < endOffset; offset += primePower) 
        	{
        	   logs[offset.get_ui()] -= primeLog;
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
    
    Number result(left.divMod(right));
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
