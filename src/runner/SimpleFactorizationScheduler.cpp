#include "SimpleFactorizationScheduler.h"
#include "factorization_montecarlo/FactorizationElf.h"

using namespace std;

SimpleFactorizationScheduler::SimpleFactorizationScheduler(const function<ElfPointer()>& factory) :
        FactorizationScheduler(factory)
{
}

void SimpleFactorizationScheduler::doDispatch()
{
    BigIntPair factors = elf().factorize(number);
    p = factors.first;
    q = factors.second;
}
