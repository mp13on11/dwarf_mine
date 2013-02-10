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
    using namespace std::placeholders;
    return QuadraticSieveHelper::factor(number, bind(&QuadraticSieveElf::sieve, &elf(), _1, _2, _3));
        //{
            //return elf()->sieve(relations
        //}
    //);
}
