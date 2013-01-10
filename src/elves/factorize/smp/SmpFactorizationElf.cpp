#include "SmpFactorizationElf.h"

#include <ctime>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

SmpFactorizationElf::SmpFactorizationElf() :
        distribution(), engine(time(NULL)), generator(bind(distribution, engine))
{
}

pair<BigInt, BigInt> SmpFactorizationElf::factorize(const BigInt& m)
{
    remainders.clear();

    while (true)
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
                    BigInt p = a + b;

                    if (p != BigInt::ONE && p != m)
                        return pair<BigInt, BigInt>(a - b, a + b);
                }
            }
            else
            {
                if (bSquared - aSquared == m)
                {
                    BigInt p = a + b;

                    if (p != BigInt::ONE && p != m)
                        return pair<BigInt, BigInt>(b - a, a + b);
                }
            }
        }

        remainders.insert(pair<BigInt, BigInt>(remainder, a));
    }
}

BigInt SmpFactorizationElf::generateRandomNumberSmallerThan(const BigInt& number) const
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
