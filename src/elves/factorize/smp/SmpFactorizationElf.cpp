#include "SequentialFactorizer.h"
#include "SmpFactorizationElf.h"

using namespace std;

pair<BigInt, BigInt> SmpFactorizationElf::factorize(const BigInt& m)
{
    SequentialFactorizer factorizer(m);

    factorizer.run();

    return factorizer.result();
}
