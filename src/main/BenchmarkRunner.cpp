
#include <mpi.h>
#include "BenchmarkRunner.h"

using namespace std;

const int MASTER = 0;

BenchmarkRunner::BenchmarkRunner(size_t iterations, IElf* elf)
    : _iterations(iterations), _elf(elf)
{
    
}

BenchmarkRunner::~BenchmarkRunner()
{
    delete _elf;
}

chrono::microseconds BenchmarkRunner::measureCall(int rank)
{
    typedef chrono::high_resolution_clock clock;
    clock::time_point before = clock::now();
    if (rank == MASTER)
    {
        _elf->distributeAndProcess();
    }
    else
    {
        _elf->process();
    }
    return clock::now() - before;
}

void BenchmarkRunner::runBenchmark()
{
    int rank = MPI::COMM_WORLD.Get_rank();
    int devices = MPI::COMM_WORLD.Get_size();
    for (int device = 0; device < devices; ++device)
    {
        chrono::microseconds sum = chrono::microseconds(0);
        for (size_t i = 0; i < _iterations; ++i)
        {
            sum += measureCall(rank);
        }
        _measurements.push_back(sum / _iterations);
    }
}
    

BenchmarkResult BenchmarkRunner::getResults()
{
    BenchmarkResult result;
    for (size_t i = 0; i < _measurements.size(); ++i)
    {
        result[i] = _measurements[i].count();
    }
    return result;
}