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

#include <gmpxx.h>
#include <gmp.h>
#include <cmath>
#include <algorithm>
#include <utility>
#include <memory>

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
