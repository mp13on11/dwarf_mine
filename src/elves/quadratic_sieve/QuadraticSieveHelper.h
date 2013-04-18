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

#include "common-factorization/BigInt.h"
#include "Types.h"
#include "Relation.h"

#include <functional>
#include <vector>
#include <list>

class PrimeFactorization;

typedef std::vector<BigInt> SmoothSquareList;
typedef std::vector<smallPrime_t> FactorBase;
typedef std::vector<Relation> Relations;
typedef std::function<std::vector<BigInt>(
    const BigInt& start,
    const BigInt& end,
    const BigInt& number,
    const FactorBase& factorBase
)> SieveSmoothSquaresCallback;

namespace QuadraticSieveHelper
{
    extern const std::pair<BigInt,BigInt> TRIVIAL_FACTORS;

    std::pair<BigInt, BigInt> factor(const BigInt& number, SieveSmoothSquaresCallback sieveCallback);

    BigInt rootModPrime(const BigInt& n, const BigInt& primeMod);
    BigInt liftRoot(const BigInt& root, const BigInt& a, const BigInt& p, uint32_t power);
    std::vector<BigInt> liftRoots(const std::vector<BigInt>& roots, const BigInt& a, const BigInt& prime, uint32_t nextPower);
    std::vector<BigInt> squareRootsModPrimePower(const BigInt& a, const BigInt& prime, uint32_t power = 1);

    FactorBase createFactorBase(size_t numberOfPrimes);
    bool isNonTrivial(const std::pair<BigInt, BigInt>& pair, const BigInt& number);
    std::pair<BigInt,BigInt> factorsFromCongruence(const BigInt& a, const BigInt& b, const BigInt& number);

    std::pair<BigInt,BigInt> searchForRandomCongruence(const FactorBase& factorBase, const BigInt& number, size_t times, const Relations& relations);
    PrimeFactorization factorizeOverBase(const BigInt& number, const FactorBase& factorBase);
    void performGaussianElimination(Relations& relations);
}
