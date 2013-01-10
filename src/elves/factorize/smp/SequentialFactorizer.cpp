#include "SequentialFactorizer.h"

#include <vector>

using namespace std;

SequentialFactorizer::SequentialFactorizer(const BigInt& number, const SmpFactorizationElf& e) :
        elf(e), m(number), p(0), q(0), distribution(), engine(e.randomSeed()),
        generator(bind(distribution, engine))
{
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

                    if (tmp == BigInt::ONE || tmp == m)
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

                    if (tmp == BigInt::ONE || tmp == m)
                        continue;

                    p = tmp;
                    q = b - a;

                    return;
                }
            }
        }

        remainders.insert(pair<BigInt, BigInt>(remainder, a));
    }
}

BigInt SequentialFactorizer::generateRandomNumberSmallerThan(const BigInt& number) const
{
    BigInt result;
    size_t maximumItems = number.buffer().size();

    if (maximumItems == 1)
    {
        uint32_t n = number.buffer().back();
        return BigInt(generator() % n);
    }

    do
    {
        size_t actualItems = generator() % maximumItems + 1;
        vector<uint32_t> items;

        for (size_t i=0; i<actualItems; i++)
        {
            items.push_back(generator());
        }

        result = BigInt(items);
    }
    while (result >= number);

    return result;
}
