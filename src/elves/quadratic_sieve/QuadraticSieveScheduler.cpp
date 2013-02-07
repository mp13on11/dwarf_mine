#include "smp/QuadraticSieve.h"
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
    number = 1089911;
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
    tie(p, q) = factor();
}

pair<BigInt, BigInt> QuadraticSieveScheduler::factor()
{
    int factorBaseSize = (int)exp(0.5*sqrt(log(number)*log(log(number))));
    cout << "factorBaseSize" << factorBaseSize << endl;
    auto factorBase = QuadraticSieveHelper::createFactorBase(factorBaseSize);

    // sieve
    cout << "sieving relations ..." << endl;
    vector<Relation> relations;
    pair<BigInt, BigInt> factors = elf().sieve(relations, factorBase, number);
    if(QuadraticSieveHelper::isNonTrivial(factors, number))
        return factors;

    cout << "found " << relations.size() << " relations" << endl;

    // bring relations into lower diagonal form
    cout << "performing gaussian elimination ..." << endl;
    QuadraticSieveHelper::performGaussianElimination(relations);

    cout << "combining random congruences ..." << endl;

    return QuadraticSieveHelper::searchForRandomCongruence(factorBase, number, 100, relations);
}
