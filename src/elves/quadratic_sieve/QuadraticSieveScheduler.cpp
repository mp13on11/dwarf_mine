#include "QuadraticSieveScheduler.h"
#include "QuadraticSieveElf.h"
#include "common/ProblemStatement.h"

using namespace std;

QuadraticSieveScheduler::QuadraticSieveScheduler(const std::function<ElfPointer()>& factory) :
    SchedulerTemplate(factory)
{

}

void QuadraticSieveScheduler::provideData(istream& input)
{
    input >> number;

    if (input.fail())
        throw runtime_error("Failed to read BigInt from input stream in " __FILE__);
}

void QuadraticSieveScheduler::outputData(ostream& output)
{
    output << p << endl;
    output << q << endl;
}

void QuadraticSieveScheduler::generateData(const DataGenerationParameters&)
{

}

bool QuadraticSieveScheduler::hasData() const
{
    return number != 0;
}

void QuadraticSieveScheduler::doSimpleDispatch()
{

}

void QuadraticSieveScheduler::doDispatch()
{

}

/*
pair<BigInt, BigInt> factorize()
{
    int factorBaseSize = (int)exp(0.5*sqrt(log(n)*log(log(n))));
    cout << "factorBaseSize" << factorBaseSize << endl;
    createFactorBase(factorBaseSize);

    // sieve
    cout << "sieving relations ..." << endl;
    pair<BigInt, BigInt> factors = sieve();
    if(isNonTrivial(factors))
        return factors;

    cout << "found " << relations.size() << " relations" << endl;

    // bring relations into lower diagonal form
    cout << "performing gaussian elimination ..." << endl;
    performGaussianElimination();

    cout << "combining random congruences ..." << endl;

    return searchForRandomCongruence(100);
}*/
