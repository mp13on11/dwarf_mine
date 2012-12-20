
#include <mpi.h>
#include <vector>
#include <memory>
#include "Elf.h"
#include "BenchmarkRunner.h"
#include "Scheduler.h"

using namespace std;

const int MASTER = 0; // TODO: Put in one common .h file
const size_t WARMUP_ITERATIONS = 50;

BenchmarkRunner::BenchmarkRunner(size_t iterations)
    : _iterations(iterations), 
      _devices(MPI::COMM_WORLD.Get_size())
{
    
}

std::chrono::microseconds BenchmarkRunner::measureCall(ProblemStatement& statement, Scheduler& scheduler) {
    typedef chrono::high_resolution_clock clock;
    clock::time_point before = clock::now();
    scheduler.dispatch(statement);
    return clock::now() - before;
}

void BenchmarkRunner::benchmarkDevice(DeviceId device, ProblemStatement& statement, Scheduler& scheduler) 
{
    for (size_t i = 0; i < WARMUP_ITERATIONS; ++i)
    {
        measureCall(statement, scheduler);
    }
    chrono::microseconds sum(0);
    for (size_t i = 0; i < _iterations; ++i)
    {
        sum += measureCall(statement, scheduler);
    }
    if (MPI::COMM_WORLD.Get_rank() == MASTER)
    {
        m_results[device] = (sum / _iterations).count();
    }
}

void BenchmarkRunner::runBenchmark(ProblemStatement& statement, const ElfFactory& factory)
{
    unique_ptr<Scheduler> scheduler = factory.createScheduler();
   
    unique_ptr<Elf> elf = factory.createElf();
    scheduler->setElf(elf.get());
    
    if (MPI::COMM_WORLD.Get_rank() == MASTER)
    {
        for (NodeId device = 0; device < _devices; ++device)
        {
            scheduler->setNodeset(device);
            benchmarkDevice(device, statement, *scheduler);
        }
    }
    else
    {
        benchmarkDevice(MASTER, statement, *scheduler);
    }
}
    

BenchmarkResult BenchmarkRunner::getResults()
{
    return m_results;
}
