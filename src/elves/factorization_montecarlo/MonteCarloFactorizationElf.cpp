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

#include "SequentialFactorizer.h"
#include "MonteCarloFactorizationElf.h"

#include <ctime>
#include <omp.h>

using namespace std;

MonteCarloFactorizationElf::MonteCarloFactorizationElf() :
        finished(false)
{
}

pair<BigInt, BigInt> MonteCarloFactorizationElf::factor(const BigInt& m)
{
    BigInt p, q;

    #pragma omp parallel shared(p, q)
    {
        SequentialFactorizer factorizer(m, *this);

        factorizer.run();
        finished = true;

        pair<BigInt, BigInt> result = factorizer.result();

        #pragma omp critical
        if (result.first != 0 && result.second != 0)
        {
            tie(p, q) = result;
        }
    }

    return pair<BigInt, BigInt>(p, q);
}

void MonteCarloFactorizationElf::stop()
{
    finished = true;
}

size_t MonteCarloFactorizationElf::randomSeed() const
{
    return time(NULL) * (omp_get_thread_num() + 1); // TODO: do something sophisticated
