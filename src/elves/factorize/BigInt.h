#pragma once

#include <gmpxx.h>
#include <gmp.h>
#include <cmath>
#include <algorithm>

typedef mpz_class BigInt;

namespace std
{
    inline BigInt min(const BigInt& a, const BigInt& b)
    {
        return (a > b) ? b : a;
    }
}

// Computes the binary logarithm with 32bit precision
inline double lb(const BigInt& x)
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

// Computes the logarithm with 32bit precision
inline double log(const BigInt& x)
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
    return log(r.get_d()) + overbits*log(2);
}

inline uint32_t lb_scaled(const BigInt& x, uint32_t maxBits)
{
    uint32_t maxLogBits = (uint32_t)ceil(log(maxBits+1)/log(2));
    uint32_t scale_shift = 32 - maxLogBits;
    return (uint32_t)(lb(x) * (1 << scale_shift));
}



// Computes the logarithm of base 2^(1/22)
// Thus the binary logarithm scaled by 2^22,
// so that all BigInts < 2^1024 will be mapped
// to the entire uint32_t range
// yielding the maximum precision after rounding
inline uint32_t log_2_22(const BigInt& x)
{
    //return (uint32_t)(lb(x) * (1 << 22));
    return lb_scaled(x, 1023);
}

inline BigInt absdiff(const BigInt& a, const BigInt& b)
{
    return (a>b)?(a-b):(b-a);
}

inline BigInt gcd(const BigInt& a, const BigInt& b)
{
    BigInt result;
    mpz_gcd(result.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t());
    return result;
}
