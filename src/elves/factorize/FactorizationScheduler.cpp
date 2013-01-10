#include "FactorizationElf.h"
#include "FactorizationScheduler.h"
#include "main/ProblemStatement.h"

#include <iostream>
#include <mpi.h>
#include <stdexcept>
#include <vector>

using namespace std;

FactorizationScheduler::FactorizationScheduler() :
    Scheduler()
{
}

FactorizationScheduler::FactorizationScheduler(const BenchmarkResult& result) :
    Scheduler(result)
{
}

void FactorizationScheduler::provideData(ProblemStatement& statement)
{
    *(statement.input) >> number;

    if (statement.input->fail())
        throw runtime_error("Failed to read BigInt from ProblemStatement in " __FILE__);
}

void FactorizationScheduler::outputData(ProblemStatement& statement)
{
    *(statement.output) << a << endl;
    *(statement.output) << b << endl;
}

void FactorizationScheduler::doDispatch()
{
    distributeNumber();
    factorizeNumber();
}

bool FactorizationScheduler::hasData()
{
    return number != 0;
}

void FactorizationScheduler::distributeNumber()
{
    if (MpiHelper::isMaster())
    {
        vector<uint32_t> items = number.buffer();
        unsigned long size = items.size();
        MPI::COMM_WORLD.Bcast(&size, 1, MPI::UNSIGNED_LONG, MpiHelper::MASTER);
        MPI::COMM_WORLD.Bcast(items.data(), size, MPI::UNSIGNED, MpiHelper::MASTER);
    }
    else
    {
        unsigned long size = 0;
        MPI::COMM_WORLD.Bcast(&size, 1, MPI::UNSIGNED_LONG, MpiHelper::MASTER);
        uint32_t *items = new uint32_t[size];
        MPI::COMM_WORLD.Bcast(items, size, MPI::UNSIGNED, MpiHelper::MASTER);

        number = BigInt(vector<uint32_t>(items, items + size));
    }
}

void FactorizationScheduler::factorizeNumber()
{
    FactorizationElf* factorizer = dynamic_cast<FactorizationElf*>(elf);

    if (factorizer == nullptr)
        return;

    pair<BigInt, BigInt> result = factorizer->factorize(number);
    a = result.first;
    b = result.second;

    cout << number << " = " << a << " * " << b << endl;
}
