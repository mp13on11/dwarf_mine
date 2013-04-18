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

#include "Types.h"
#include "common-factorization/BigInt.h"
#include "SparseVector.h"
#include <list>
#include <utility>
#include <iosfwd>

class PrimeFactorization
{
public:
    bool empty() const;
    PrimeFactorization combine(const PrimeFactorization& other) const;
    PrimeFactorization sqrt() const;
    BigInt multiply() const;
    SparseVector<smallPrime_t> oddPrimePowers() const;
    void add(const smallPrime_t& prime, uint32_t power);

    void print(std::ostream& stream) const;

private:
    std::list<std::pair<smallPrime_t, uint32_t>> primePowers;
};

inline PrimeFactorization sqrt(const PrimeFactorization& p)
{
    return p.sqrt();
}

