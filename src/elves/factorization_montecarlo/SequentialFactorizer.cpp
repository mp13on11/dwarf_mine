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

#include <vector>

using namespace std;

SequentialFactorizer::SequentialFactorizer(const BigInt& number, const MonteCarloFactorizationElf& e) :
        elf(e), m(number), p(0), q(0), generator(gmp_randinit_mt)
{
    generator.seed(elf.randomSeed());
}

void SequentialFactorizer::run()
{
    while (!elf.finished && p == 0 && q == 0)
    {
        BigInt a = generateRandomNumberSmallerThan(m);
        BigInt aSquared = a * a;
        BigInt remainder = aSquared % m;

        iterator_pair range = remainders.equal_range(remainder);

        for (iterator it = range.first; it != range.second; it++)
        {
            const BigInt& b = it->second;
            BigInt bSquared = b * b;

            if (a > b)
            {
                if (aSquared - bSquared == m)
                {
                    BigInt tmp = a + b;

                    if (tmp == m)
                        continue;

                    p = tmp;
                    q = a - b;

                    return;
                }
            }
            else
            {
                if (bSquared - aSquared == m)
                {
                    BigInt tmp = a + b;

                    if (tmp == m)
                        continue;

                    p = tmp;
                    q = b - a;

                    return;
                }
            }
        }

        remainders.insert({remainder, a});
    }
}

BigInt SequentialFactorizer::generateRandomNumberSmallerThan(const BigInt& number)
{
    return generator.get_z_range(number);
