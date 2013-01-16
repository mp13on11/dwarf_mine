#pragma once

#include <gmpxx.h>
#include <gmp.h>
#include <cmath>

typedef mpz_class BigInt;

// Computes the binary logarithm with 32bit precision
inline double lb(BigInt& x)
{
    auto bits = mpz_sizeinbase(x.get_mpz_t(), 2);
    mp_bitcnt_t overbits;
    if(bits > 32)
    {
        overbits = bits - 32;
    }
    else
    {
        overbits = 0;
    }
    BigInt r = (x >> overbits);
    return log(r.get_d())/log(2) + overbits;
}

// Computes the logarithm of base 2^(1/22)
// Thus the binary logarithm scaled by 2^22,
// so that all BigInts < 2^1024 will be mapped 
// to the entire uint32_t range
// yielding the maximum precision after rounding
inline uint32_t log_2_22(BigInt& x)
{
    return (uint32_t)(lb(x) * (1 << 22));
}
