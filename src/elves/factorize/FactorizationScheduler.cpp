#include "FactorizationElf.h"
#include "FactorizationScheduler.h"
#include "main/ProblemStatement.h"

#include <memory>
#include <mpi.h>
#include <stdexcept>
#include <vector>

using namespace std;
using namespace std::chrono;

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

    future<BigIntPair> f = async(launch::async, [&]{
            return factorizeNumber();
        });

    int rank = distributeFinishedStateRegularly(f);
    stopFactorization();
    sendResultToMaster(rank, f);
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
        unique_ptr<uint32_t[]> items(new uint32_t[size]);
        MPI::COMM_WORLD.Bcast(items.get(), size, MPI::UNSIGNED, MpiHelper::MASTER);

        number = BigInt(vector<uint32_t>(items.get(), items.get() + size));
    }
}

FactorizationScheduler::BigIntPair FactorizationScheduler::factorizeNumber()
{
    FactorizationElf* factorizer = dynamic_cast<FactorizationElf*>(elf);

    if (factorizer == nullptr)
        return BigIntPair(0, 0);

    BigIntPair result = factorizer->factorize(number);

    return result;
}

void FactorizationScheduler::sendResultToMaster(int rank, future<BigIntPair>& f)
{
    if (!MpiHelper::isMaster() && rank != MpiHelper::rank())
        return;

    if (rank == MpiHelper::rank())
    {
        BigIntPair pair = f.get();
        a = pair.first;
        b = pair.second;
    }

    if (MpiHelper::isMaster())
    {
        unsigned long sizes[] = { 0, 0 };
        MPI::COMM_WORLD.Recv(sizes, 2, MPI::UNSIGNED_LONG, rank, 1);

        unique_ptr<uint32_t[]> first(new uint32_t[sizes[0]]);
        unique_ptr<uint32_t[]> second(new uint32_t[sizes[1]]);

        MPI::COMM_WORLD.Recv(first.get(), sizes[0], MPI::UNSIGNED, rank, 2);
        MPI::COMM_WORLD.Recv(second.get(), sizes[1], MPI::UNSIGNED, rank, 3);

        a = BigInt(vector<uint32_t>(first.get(), first.get() + sizes[0]));
        b = BigInt(vector<uint32_t>(second.get(), second.get() + sizes[1]));
    }
    else
    {
        vector<uint32_t> first = a.buffer();
        vector<uint32_t> second = b.buffer();
        unsigned long sizes[] = { first.size(), second.size() };

        MPI::COMM_WORLD.Send(sizes, 2, MPI::UNSIGNED_LONG, MpiHelper::MASTER, 1);
        MPI::COMM_WORLD.Send(first.data(), first.size(), MPI::UNSIGNED, MpiHelper::MASTER, 2);
        MPI::COMM_WORLD.Send(second.data(), second.size(), MPI::UNSIGNED, MpiHelper::MASTER, 3);
    }
}

int FactorizationScheduler::distributeFinishedStateRegularly(future<BigIntPair>& f) const
{
    unique_ptr<bool[]> states(new bool[MpiHelper::numberOfNodes()]);

    while (true)
    {
        bool finished = f.wait_for(seconds(5)) == future_status::ready;

        MPI::COMM_WORLD.Allgather(
                &finished, 1, MPI::BOOL,
                states.get(), 1, MPI::BOOL
            );

        for (size_t i=0; i<MpiHelper::numberOfNodes(); i++)
        {
            if (states[i])
                return i;
        }
    }
}

void FactorizationScheduler::stopFactorization()
{
    FactorizationElf* factorizer = dynamic_cast<FactorizationElf*>(elf);

    if (factorizer == nullptr)
        return;

    factorizer->stop();
}
