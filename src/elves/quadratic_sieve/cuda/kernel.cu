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
    Number newEnd = newStart + Number(NUMBERS_PER_THREAD);
    
    if (newStart > *end) 
    {
        return;
    }
    
   
    if (index == 0) {
    printf("start: %d\n", start->get_ui());
    printf("newEnd: %d\n", newEnd.get_ui());
    printf("newStart: %d\n", newStart.get_ui());
    }
    
    //printf("asdfasdlf jasldfkaljsdfka jsdf \n\n\n");

    for (int i=0; i<factorBaseSize; ++i)
    {
    	Number prime(factorBase[i]);
    	Number primePower(prime);
    	
        //printf("primePower %d < number %d\n", primePower.get_ui(), number->get_ui());
        //printf("primePower < number %d\n", primePower > *number);
        
    	while (primePower < *number) {
    	   
    	    printf("testing primePower %d\n", primePower.get_ui());    
    	    	    
    	    int timesAdvanced = 0;
        	
        	printf("about to print\n");
        	Number h = newStart % primePower;
        	printf("h: %d", h.get_ui());
        	//while (timesAdvanced <= NUMBERS_PER_THREAD && !(newStart % primePower).isZero() && newStart <= *end) 
        	while (true)
        	{
        	   
        	   printf("advanced\n");
        	   newStart += 1;
        	   ++timesAdvanced;
        	   if (timesAdvanced >= NUMBERS_PER_THREAD) break;
        	   //rest = newStart % primePower;
        	} 
        	
        	printf("after while loop\n\n\n");
        	
        	for (; newStart < newEnd; newStart += primePower) 
        	{
        	   logs[newStart.get_ui()] -= log_2_22(primePower);
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
