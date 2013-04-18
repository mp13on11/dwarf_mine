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

#include "CudaQuadraticSieveElf.h"
#include "Factorize.h"
#include "common/Utils.h"
#include "common-factorization/BigInt.h"
#include <cuda-utils/Memory.h>
#include "NumberHelper.h"
#include "KernelWrapper.h"
#include <iostream>

#include <array>
#include <algorithm>

using namespace std;

vector<BigInt> CudaQuadraticSieveElf::sieveSmoothSquares(
        const BigInt& start,
        const BigInt& end,
        const BigInt& number,
        const FactorBase& factorBase
)
{
    BigInt intervalLength = (end-start);

    size_t blockSize = intervalLength.get_ui();

    vector<uint32_t> logs(blockSize+1);
    BigInt x, remainder;

    // init field with logarithm
    x = start;
    for(uint32_t i=0; i<=blockSize; i++, x++)
    {
        remainder = (x*x) % number;
        logs[i] = log_2_22(remainder);
    }

    CudaUtils::Memory<uint32_t> logs_d(logs.size());
    logs_d.transferFrom(logs.data());
    CudaUtils::Memory<uint32_t> factorBase_d(factorBase.size());
    factorBase_d.transferFrom(factorBase.data());

    array<uint32_t, 10> number_d;
    number_d.fill(0);
    mpz_export((void*)number_d.data(), 0, -1, sizeof(uint32_t), 0, 0, number.get_mpz_t());

    array<uint32_t, 10> start_d;
    start_d.fill(0);
    mpz_export((void*)start_d.data(), 0, -1, sizeof(uint32_t), 0, 0, start.get_mpz_t());

    array<uint32_t, 10> end_d;
    end_d.fill(0);
    mpz_export((void*)end_d.data(), 0, -1, sizeof(uint32_t), 0, 0, end.get_mpz_t());

    megaWrapper(number_d.data(), logs_d.get(), factorBase_d.get(), factorBase.size(), start_d.data(), end_d.data(), blockSize);
    vector<uint32_t> newLogs(blockSize+1);
    logs_d.transferTo(newLogs.data());

    vector<BigInt> result;
    uint32_t logTreshold = (int)(lb(number));
    for(uint32_t i=0; i<=blockSize; i++)
    {
        if(newLogs[i] < logTreshold) // probable smooth
        {
            result.emplace_back(start+i);
        }
    }
    return result;
