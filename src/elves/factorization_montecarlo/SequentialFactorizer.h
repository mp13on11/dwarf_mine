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

#include "elves/common-factorization/BigInt.h"
#include "MonteCarloFactorizationElf.h"

#include <functional>
#include <map>

class SequentialFactorizer
{
public:
    SequentialFactorizer(const BigInt& m, const MonteCarloFactorizationElf& elf);

    void run();
    std::pair<BigInt, BigInt> result() const;

private:
    const MonteCarloFactorizationElf& elf;
    BigInt m, p, q;
    std::multimap<BigInt, BigInt> remainders;
    gmp_randclass generator;

    typedef std::multimap<BigInt, BigInt>::iterator iterator;
    typedef std::pair<iterator, iterator> iterator_pair;

    BigInt generateRandomNumberSmallerThan(const BigInt& n);
};

inline std::pair<BigInt, BigInt> SequentialFactorizer::result() const
{
    return std::pair<BigInt, BigInt>(p, q);
}
