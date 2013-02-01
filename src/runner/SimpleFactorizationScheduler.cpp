#include "SimpleFactorizationScheduler.h"
#include "factorization_montecarlo/MonteCarloFactorizationElf.h"

using namespace std;

SimpleFactorizationScheduler::SimpleFactorizationScheduler(const function<ElfPointer()>& factory) :
        FactorizationScheduler(factory)
{
}

void SimpleFactorizationScheduler::doDispatch()
{
    BigIntPair factors = elf().factor(number);
    p = factors.first;
    q = factors.second;
}
