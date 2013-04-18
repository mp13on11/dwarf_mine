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

#include <elves/cuda-utils/Memory.h>
#include "Factorize.h"
#include <vector>
#include <array>

class NumberHelper
{
public:
    static CudaUtils::Memory<uint32_t> BigIntToNumber(const BigInt& b)
    {
        NumData numberData;
        memset(numberData, 0, sizeof(uint32_t) * NUM_FIELDS);
        mpz_export(numberData, nullptr, -1, sizeof(uint32_t), 0, 0, b.get_mpz_t());

        CudaUtils::Memory<uint32_t> number_d(NUM_FIELDS);
        number_d.transferFrom(numberData);
        return number_d;
    }

    static CudaUtils::Memory<uint32_t> BigIntsToNumbers(const std::vector<BigInt>& bigints)
    {
        std::vector<uint32_t> numbersData;
        for (auto bigintIter = bigints.begin(); bigintIter != bigints.end(); ++bigintIter)
        {
            NumData numberData;
            memset(numberData, 0, sizeof(uint32_t) * NUM_FIELDS);
            mpz_export(numberData, nullptr, -1, sizeof(uint32_t), 0, 0, (*bigintIter).get_mpz_t());

            for (int i = 0; i < NUM_FIELDS; ++i)
                numbersData.push_back(numberData[i]);
        }

        CudaUtils::Memory<uint32_t> number_d(bigints.size()*NUM_FIELDS);
        number_d.transferFrom(&numbersData[0]);
        return number_d;
    }

    static std::vector<uint32_t> NumbersToUis(const CudaUtils::Memory<uint32_t>& number_d)
    {
        std::vector<uint32_t> result;
        std::vector<uint32_t> numbers(number_d.numberOfElements());
        number_d.transferTo(&numbers[0]);

        for (size_t i = 0; i < number_d.numberOfElements(); i += NUM_FIELDS)
        {
            result.push_back(numbers[i]);     
        }

        return result;
    }

    static BigInt NumberToBigInt(const CudaUtils::Memory<uint32_t>& number_d)
    {
        NumData outputData;
        number_d.transferTo(outputData);

        BigInt mpzResult;
        mpz_import(mpzResult.get_mpz_t(), NUM_FIELDS, -1, sizeof(uint32_t), 0, 0, outputData);
        return mpzResult;
    }
};
