#include "SequentialFactorizer.h"
#include "MonteCarloFactorizationElf.h"
#include "common/MpiHelper.h"

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
            p = result.first;
            q = result.second;
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
    return time(NULL) * (omp_get_thread_num() + 1) + MpiHelper::rank();
}
