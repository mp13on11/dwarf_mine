#include "Number.cuh"
#include <cuda.h>
#include <stdio.h>

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
            high = (n.fields[i] << o) & (0xFFFFFFFF << o);
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

//struct PrimePower
//{
//    uint32_t prime;
//    uint32_t power;
//
//    __device__ PrimePower() : prime((uint32_t)0), power((uint32_t)0)
//    {
//    }
//
//    __device__ PrimePower(uint32_t p, uint32_t pow) : prime(p), power(pow)
//    {
//    }
//};
//
//struct PrimeFactorization
//{
//    PrimePower primePowers[10];
//    int index;
//
//    __device__ PrimeFactorization() : index(0)
//    {
//    }
//
//    __device__ void add(uint32_t prime, uint32_t power)
//    {
//        primePowers[index].prime = prime;
//        primePowers[index].power = power;
//        ++index;
//    }
//};
//
//
//__device__ PrimeFactorization factorizeOverBase(const Number& n, uint32_t* factorBase, int factorBaseSize)
//{
//    PrimeFactorization result;
//    Number x(n);
//    Number remainder;
//    for (int i = 0; i < factorBaseSize && x > 1; ++i)
//    {
//        Number bigPrime = Number(factorBase[i]);
//        uint32_t smallPrime = factorBase[i];
//        uint32_t power = 0;
//        do
//        {
//            Number remainder = n % bigPrime;
//        } while (remainder.isZero());
//
//        //if (power > 0)
//    }
//
//    return result;
//
//}

__global__ void sieveIntervalKernel(PNumData pn, uint32_t* logs, uint32_t* rootsModPrime, uint32_t* factorBase, int factorBaseSize, PNumData pStart, PNumData pEnd)
{
    //printf("log_2_22Log: %d\n", log_2_22(Number(pn)));
    //printf("binaryLog: %f\n", lb(Number(pn)));
    Number start(pStart);
    Number end(pEnd);
    Number n(pn);
    uint32_t intervalLength = (end - start).get_ui();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= factorBaseSize) return;

    Number bigPrime(factorBase[idx]);
    uint32_t prime = bigPrime.get_ui();
    Number root = rootsModPrime[idx*NUM_FIELDS];
    
    if (root.isZero()) return;

    for (int z = 0; z < 2; ++z)
    {
        Number offset = (bigPrime + root - (start % bigPrime)) % bigPrime;
        uint32_t primeLog = log_2_22(bigPrime);
        for (int i = 0; i <= intervalLength; i += prime)
        {
            if (idx == 0)
                printf("iter: %d\n", i);
            logs[i] -= primeLog;
        } 

        if (bigPrime-root == root)
            break;
        else
            root = bigPrime - root;
    }

    //uint32_t logThreshold = lb(n);
    //for (int i = 0; i <= intervalLength; ++i) 
    //{
    //    if (logs[i] < logThreshold)
    //    {
    //        Number x = start + Number(i);
    //        Number remainder = (x*x) % n;

    //    }
    //}

}

