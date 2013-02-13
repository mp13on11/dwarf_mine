#include "SmpQuadraticSieveElf.h"
#include <common/Utils.h>

#include <algorithm>

using namespace std;

// returns a list of numbers, whose quadratic residues are (probable) smooth
// over the factor base
vector<BigInt> SmpQuadraticSieveElf::sieveSmoothSquares(const BigInt& start, const BigInt& end, const BigInt& number, const FactorBase& factorBase)
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
