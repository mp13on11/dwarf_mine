#include "FactorizationTest.h"
#include "Utilities.h"
#include <main/ElfFactory.h>
#include <elves/factorize/BigInt.h>
#include <elves/factorize/cuda/CudaFactorizationElf.h>

#include <memory>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace std;
using namespace chrono;

typedef vector<uint64_t> PrimeList;

const PrimeList SMALL_PRIMES = { 2, 3, 5, 7, 11, 13, 17, 19, 23 };
const PrimeList PRIMES_BELOW_100 = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97 };

const PrimeList SIEVE_PARAMS = { 37, 100, 500, 1000 };

INSTANTIATE_TEST_CASE_P(
    MultiplePlatforms,
    FactorizationTest,
    testing::Values("cuda"));

PrimeList goldSieve(uint64_t threshold)
{
    PrimeList allNumbers(threshold-1);

    for (uint64_t i=2; i<=threshold; ++i)
        allNumbers[i-2] = i;

    uint64_t candidate = 2;
    while (candidate < ceil(sqrt(threshold)))
    {
        auto toRemove = candidate*candidate;

        for (; toRemove <= threshold; toRemove += candidate)
            allNumbers[toRemove - 2] = 0;

        --candidate;
        for (; allNumbers[candidate] == 0; ++candidate);
        candidate += 2;
    }

    auto newIter = remove_if(allNumbers.begin(), allNumbers.end(), [](uint64_t elem) { return elem == 0; });
    return PrimeList(allNumbers.begin(), newIter);
}

TEST_P(FactorizationTest, FactorizesSimpleCompositesTest)
{

    auto result = elf->factorize(2 * 6);
    auto num = result.second;

    ASSERT_EQ(BigInt("18446744073709551616"), num);
    //cout << result.second << endl;

/*
    vector<uint32_t> primesBut2(SMALL_PRIMES.begin()+1, SMALL_PRIMES.end());
    uint32_t two = 2;

    for (uint32_t prime : primesBut2)
    {
        BigInt simple(two * prime);
        auto result = elf->factorize(simple);

        if (result.first > result.second)
            std::swap(result.first, result.second);

        ASSERT_EQ(two, result.first.getUint32Value());
        ASSERT_EQ(prime, result.second.getUint32Value());
    }*/
}

void FactorizationTest::SetUp()
{
    //auto factory = createElfFactory(GetParam(), "factorize");
    //auto theElf = factory->createElf();
    elf.reset(new CudaFactorizationElf());
}
