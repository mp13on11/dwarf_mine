#include "MonteCarloFactorizationElf.h"
#include "FactorizationScheduler.h"
#include "common/ProblemStatement.h"

#include <memory>
#include <mpi.h>
#include <stdexcept>
#include <vector>

using namespace std;
using namespace std::chrono;

const BigInt DEFAULT_PRODUCT(1649);

FactorizationScheduler::FactorizationScheduler(const function<ElfPointer()>& factory) :
    SchedulerTemplate(factory)
{
}

void FactorizationScheduler::doSimpleDispatch()
{
    tie(p, q) = elf().factor(number);
}

void FactorizationScheduler::provideData(istream& input)
{
    input >> number;

    if (input.fail())
        throw runtime_error("Failed to read BigInt from input stream in " __FILE__);
}

void FactorizationScheduler::outputData(ostream& output)
{
    output << p << endl;
    output << q << endl;
}

void FactorizationScheduler::generateData(const DataGenerationParameters&)
{
    number = DEFAULT_PRODUCT;
}

void FactorizationScheduler::doDispatch()
{
    distributeNumber();

    future<BigIntPair> f = async(launch::async,
        &MonteCarloFactorizationElf::factor, &elf(), number
    );

    int rank = distributeFinishedStateRegularly(f);
    elf().stop();
    sendResultToMaster(rank, f);
}

bool FactorizationScheduler::hasData() const
{
    return number != 0;
}

void FactorizationScheduler::distributeNumber()
{
    if (MpiHelper::isMaster())
    {
        string s = number.get_str();
        unsigned long size = s.length();
        MPI::COMM_WORLD.Bcast(&size, 1, MPI::UNSIGNED_LONG, MpiHelper::MASTER);
        MPI::COMM_WORLD.Bcast(
                const_cast<char*>(s.c_str()), size,
                MPI::CHAR, MpiHelper::MASTER
            );
    }
    else
    {
        unsigned long size = 0;
        MPI::COMM_WORLD.Bcast(&size, 1, MPI::UNSIGNED_LONG, MpiHelper::MASTER);
        unique_ptr<char[]> items(new char[size]);
        MPI::COMM_WORLD.Bcast(items.get(), size, MPI::CHAR, MpiHelper::MASTER);

        number = BigInt(string(items.get(), size));
    }
}

void FactorizationScheduler::sendResultToMaster(int rank, future<BigIntPair>& f)
{
    if (!MpiHelper::isMaster() && rank != MpiHelper::rank())
        return;

    if (rank == MpiHelper::rank())
    {
        BigIntPair pair = f.get();
        p = pair.first;
        q = pair.second;
    }

    // if the master found the result, nothing more to do
    if(MpiHelper::isMaster(rank))
        return;

    // a slave found the result, he has to send it to the master
    if (MpiHelper::isMaster())
    {
        unsigned long sizes[] = { 0, 0 };
        MPI::COMM_WORLD.Recv(sizes, 2, MPI::UNSIGNED_LONG, rank, 1);

        unique_ptr<char[]> first(new char[sizes[0]]);
        unique_ptr<char[]> second(new char[sizes[1]]);

        MPI::COMM_WORLD.Recv(first.get(), sizes[0], MPI::CHAR, rank, 2);
        MPI::COMM_WORLD.Recv(second.get(), sizes[1], MPI::CHAR, rank, 3);

        p = BigInt(string(first.get(), sizes[0]));
        q = BigInt(string(second.get(), sizes[1]));
    }
    else
    {
        string first = p.get_str();
        string second = q.get_str();
        unsigned long sizes[] = { first.length(), second.length() };

        MPI::COMM_WORLD.Send(sizes, 2, MPI::UNSIGNED_LONG, MpiHelper::MASTER, 1);
        MPI::COMM_WORLD.Send(first.c_str(), first.size(), MPI::CHAR, MpiHelper::MASTER, 2);
        MPI::COMM_WORLD.Send(second.c_str(), second.size(), MPI::CHAR, MpiHelper::MASTER, 3);
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

void FactorizationScheduler::doBenchmarkDispatch(NodeId /*node*/)
{
	// Pre-benchmark does not make sense for
	// this algorithm - left empty
}    
