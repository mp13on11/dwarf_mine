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
}
