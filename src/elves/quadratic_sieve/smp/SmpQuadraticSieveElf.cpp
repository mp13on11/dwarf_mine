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

#include "SmpQuadraticSieveElf.h"
#include <common/Utils.h>

#include <algorithm>
#include <future>
#include <cassert>
#include <thread>
#include <omp.h>

using namespace std;

// returns a list of numbers, whose quadratic residues are (probable) smooth
// over the factor base
vector<BigInt> smpSieveKernel(const BigInt& start, const BigInt& end, const BigInt& number, const FactorBase& factorBase)
{
    BigInt intervalLength = (end-start);
    size_t blockSize = intervalLength.get_ui();
    vector<uint32_t> logs(blockSize+1);
    BigInt x, remainder;
    uint32_t logTreshold = (int)(lb(number));

    // init field with logarithm
    x = start;

    for(uint32_t i=0; i<=blockSize; ++i, ++x)
    {
        remainder = (x*x) % number;
        logs[i] = log_2_22(remainder);
    }

    // now with prime powers
    for(const smallPrime_t& smallPrime : factorBase)
    {
        BigInt prime(smallPrime);
        uint32_t primeLog = log_2_22(prime);
        uint32_t i = 1;
        BigInt primePower = prime;
        for(; primePower < number; i++, primePower*=prime)
        {
            vector<BigInt> roots = QuadraticSieveHelper::squareRootsModPrimePower(number%primePower, prime, i);
            for(const BigInt& root : roots)
            {
                BigInt offset = (primePower + root - (start % primePower)) % primePower;
                for(BigInt j=offset; j<=blockSize; j+=primePower)
                {
                    logs[j.get_ui()] -= primeLog;
                }
            }
        }
    }

    //second scan for smooth numbers
    BigInt biggestPrime(factorBase.back());
    vector<BigInt> result;

    for(uint32_t i=0; i<=blockSize; i++)
    {
        if(logs[i] < logTreshold) // probable smooth
        {
            result.emplace_back(start+i);
        }
    }
    
    return result;
}

vector<BigInt> SmpQuadraticSieveElf::sieveSmoothSquares(const BigInt& start, const BigInt& end, const BigInt& number, const FactorBase& factorBase)
{
    // Ensure execution can be controlled via 
    // OMP_NUM_THREADS environment variable
    const int NUM_THREADS = omp_get_max_threads();

    SmoothSquareList smooths;
    vector<future<SmoothSquareList>> partialResults;
    BigInt totalLength = end - start;
    BigInt chunkSize = div_ceil(totalLength, BigInt(NUM_THREADS));

    for (int i=0; i<NUM_THREADS; ++i)
    {
        BigInt partialStart = start + chunkSize*i;
        BigInt partialEnd = min(partialStart + chunkSize, end);

        partialResults.emplace_back(std::async(
            std::launch::async,
            smpSieveKernel, partialStart, partialEnd, number, factorBase
        ));
    }

    for (auto& result : partialResults)
    {
        auto partialResult = result.get();
        smooths.insert(smooths.end(), partialResult.begin(), partialResult.end());
    }
    
    return smooths;
}
