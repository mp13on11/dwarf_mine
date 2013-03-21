#include "kernel.cuh"
#include "Number.cuh"
#include "stdio.h"


__device__ uint32_t pow(Number b, Number e, Number m)
{
    Number result = 1;
    for (int i = (NUM_FIELDS*32)-1; i>=0; --i)
    {
       result *= result;
       if (e.bitAt(i) == 1)
       {
           result *= b;
       }
       result = result % m;
    }
    return result.get_ui();
}

__device__ uint32_t cuda_pow(uint32_t b, uint32_t e)
{
    uint32_t res = 1;
    while (e != 0)
    {
        while (e & 1 != 1)
        {
            e >>= 1;
            b = (b*b);
        }
        --e;
        res = res * b;
    }
    return res;
}

__device__ int legendre_symbol(Number a, Number p)
{
    Number ls = pow(a, (p-Number(1))/Number(2), p);
    if(Number(5) < Number(2))
        printf("Is No good\n");
    printf("legendre exp: %u\n", Number(5).get_ui());
    printf("legendre exp: %u\n", Number(1).get_ui());
    printf("legendre exp: %u\n", (Number(1)+Number(1)).get_ui());
    printf("legendre exp: %u\n", (Number(5)-Number(1)).get_ui());
    printf("legendre exp: %u\n", ((Number(5)-Number(1))/Number(2)).get_ui());
    if (ls == (p - Number(1)))
        return -1;
    else 
        return ls.get_ui();
}

__device__ uint32_t rootModPrime(Number a, Number p)
{
    if (legendre_symbol(a, p) != -1)
        return 0;
    else if (a.isZero())
        return 0;
    else if (p == Number(2))
        return p.get_ui();
    else if (p % Number(4) == Number(3))
        return pow(a, (p + Number(1)) / Number(4), p);

    // Partition p-1 to s * 2^e for an odd s (i.e.
    // reduce all the powers of 2 from p-1)
    Number s = p - Number(1);
    int e = 0;
    while ((s % Number(2)).isZero())
        s /= 2;
        e += 1;

    // Find some 'n' with a legendre symbol n|p = -1.
    // Shouldn't take long.
    //
    Number n(2);
    while (legendre_symbol(n, p) != -1)
        n += 1;

    // Here be dragons!
    // Read the paper "Square roots from 1; 24, 51,
    // 10 to Dan Shanks" by Ezra Brown for more
    // information
    

    // x is a guess of the square root that gets better
    // with each iteration.
    // b is the "fudge factor" - by how much we're off
    // with the guess. The invariant x^2 = ab (mod p)
    // is maintained throughout the loop.
    // g is used for successive powers of n to update
    // both a and b
    // r is the exponent - decreases with each update
    
    uint32_t x = pow(a, (s + Number(1)) / Number(2), p);
    uint32_t b = pow(a, s, p);
    uint32_t g = pow(n, s, p);
    uint32_t r = e;

    while (true)
    {
        uint32_t t = b;
        uint32_t m = 0;
        for (; m < r; ++m)
            if (t == 1)
                break;
            t = pow(t, 2, p);

        if (m == 0)
            return x;

        uint32_t gs = pow(g, cuda_pow(2, (r - m - 1)), p);
        g = (gs * gs) % p.get_ui();
        x = (x * gs) % p.get_ui();
        b = (b * g) % p.get_ui();
        r = m;
    }
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
            //unsigned int q = ((newStart*newStart) - *number).get_ui();
        	//while (!(q % primePower.get_ui() == 0)) 
        	while (!(q % primePower).isZero()) 
        	{        	   
        	   ++timesAdvanced;
        	   newStart += 1;
        	   if ((timesAdvanced > NUMBERS_PER_THREAD) || (newStart > *end)) 
        	   {
        	       invalid = true;
        	       break;
        	   }
               //q = ((newStart * newStart) - *number).get_ui();
               q = ((newStart * newStart) - *number);
        	} 
        	
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

__global__ void testLegendreKernel(PNumData pA, PNumData pPrime, int* output)
{
    Number a(pA);
    Number p(pPrime);

    int result(legendre_symbol(a, p));
    *output = result;
}
