#include "SequentialFactorizer.h"
#include "SmpFactorizationElf.h"
#include "main/MpiHelper.h"

#include <ctime>
#include <omp.h>

using namespace std;

SmpFactorizationElf::SmpFactorizationElf() :
        finished(false)
{
}

pair<BigInt, BigInt> SmpFactorizationElf::factorize(const BigInt& m)
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

void SmpFactorizationElf::stop()
{
    finished = true;
}

size_t SmpFactorizationElf::randomSeed() const
{
    return time(NULL) * (omp_get_thread_num() + 1) + MpiHelper::rank();
}
