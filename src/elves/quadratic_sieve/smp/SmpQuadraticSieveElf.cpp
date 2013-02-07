#include "SmpQuadraticSieveElf.h"
#include <common/Utils.h>

#include <future>
#include <cassert>
#include <thread>
#include <algorithm>

using namespace std;

// returns a list of numbers, which quadratic residues are (probable) smooth over the factor base
vector<BigInt> sieveSmoothSquares(const BigInt& start, const BigInt& end, const BigInt& number, const FactorBase& factorBase)
{
    BigInt intervalLength = (end-start);
    size_t blockSize = intervalLength.get_ui();
    vector<uint32_t> logs(blockSize+1);
    BigInt x, remainder;
    uint32_t logTreshold = (int)(lb(number));

    // init field with logarithm
    x = start;
    for(uint32_t i=0; i<=blockSize; i++, x++)
    {
        remainder = (x*x) % number;
        logs[i] = log_2_22(remainder);
    }

    // now with prime powers
    cout << "starting with logarithmic sieving ..." << endl;
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

pair<BigInt, BigInt> sieveIntervalFast(const BigInt& start, const BigInt& end, vector<Relation>& relations, const FactorBase& factorBase, const BigInt& number)
{
    const size_t maxRelations = factorBase.size() + 2;

    const int NUM_THREADS = 4;

    SmoothSquares smooths;
    vector<future<SmoothSquares>> partialResults;
    BigInt totalLength = end - start;
    //BigInt chunkSize = //totalLength / NUM_THREADS;
    BigInt chunkSize = div_ceil(totalLength, BigInt(NUM_THREADS));

    for (int i=0; i<NUM_THREADS; ++i)
    {
        BigInt partialStart = start + chunkSize*i;
        BigInt partialEnd = min(partialStart + chunkSize, end);

        partialResults.emplace_back(std::async(
            std::launch::async,
            &sieveSmoothSquares, partialStart, partialEnd, number, factorBase
        ));
    }

    for (auto& result : partialResults)
    {
        auto partialResult = result.get();
        smooths.insert(smooths.end(), partialResult.begin(), partialResult.end());
    }

    for(const BigInt& x : smooths)
    {
        BigInt remainder = (x*x) % number;

        PrimeFactorization factorization = QuadraticSieveHelper::factorizeOverBase(remainder, factorBase);
        if(factorization.empty())
        {
            cerr << "false alarm !!! (should not happend)" << endl;
            continue;
        }

        Relation relation(x, factorization);

        if(relation.isPerfectCongruence())
        {
            auto factors = QuadraticSieveHelper::factorsFromCongruence(x, sqrt(factorization).multiply(), number);
            if(QuadraticSieveHelper::isNonTrivial(factors, number))
            {
                return factors;
            }
        }

        relations.push_back(relation);

        if(relations.size() >= maxRelations)
            break;
    }

    return QuadraticSieveHelper::TRIVIAL_FACTORS;
}

pair<BigInt, BigInt> SmpQuadraticSieveElf::sieve(vector<Relation>& relations, const FactorBase& factorBase, const BigInt& number)
{
    BigInt intervalSize = exp(sqrt(log(number)*log(log(number))));
    BigInt intervalStart = sqrt(number) + 1;
    BigInt intervalEnd = sqrt(number)+ 1 + intervalSize;
    intervalEnd = (sqrt(2*number) < intervalEnd) ? sqrt(2*number) : intervalEnd;

    cout << "sieving interval: " << (intervalEnd - intervalStart) << endl;
    return sieveIntervalFast(intervalStart, intervalEnd, relations, factorBase, number);
}
