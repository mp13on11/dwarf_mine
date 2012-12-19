
#include <mpi.h>
#include <vector>
#include <sstream>
#include <memory>
#include "Elf.h"
#include "BenchmarkRunner.h"

using namespace std;

const int MASTER = 0;

BenchmarkRunner::BenchmarkRunner(size_t iterations)
    : _iterations(iterations), _rank(MPI::COMM_WORLD.Get_rank()), 
      _devices(MPI::COMM_WORLD.Get_size())
{
    
}

void receive(ostream& stream, int fromRank)
{
    MPI::Status status;
    MPI::COMM_WORLD.Probe(fromRank, 0, status);
    int bufferSize = status.Get_count(MPI::CHAR);
    unique_ptr<char[]> buffer(new char[bufferSize]);
    MPI::COMM_WORLD.Recv(buffer.get(), bufferSize, MPI::CHAR, fromRank, 0);
    stream.write(buffer.get(), bufferSize);
}

void send(istream& stream, int toRank)
{
    stringstream buffer;
    buffer << stream.rdbuf();
    auto buffered = buffer.str();
    MPI::COMM_WORLD.Send(buffered.c_str(), buffered.size(), MPI::CHAR, toRank, 0);
}

chrono::microseconds BenchmarkRunner::measureCall(int targetRank, Elf& elf, const ProblemStatement& statement)
{
    typedef chrono::high_resolution_clock clock;
    clock::time_point before = clock::now();
    if (_rank == MASTER)
    {
        statement.input.clear();
        statement.input.seekg(0, ios::beg);
        stringstream output;
        if (targetRank == MASTER)
        {
            
            elf.run(statement.input, output);
        }
        else
        {
            send(statement.input, targetRank);
            receive(output, targetRank);
        }
    }
    else
    {
        receive(statement.input, MASTER);
        elf.run(statement.input, statement.output);
        send(statement.output, MASTER);
    }
    return clock::now() - before;
}

void BenchmarkRunner::benchmarkDevice(int device, Elf& elf, const ProblemStatement& statement)
{
    size_t warmupIterations = _iterations / 10;
    for (size_t i = 0; i < warmupIterations; ++i)
    {
        measureCall(device, elf, statement);
    }
    chrono::microseconds sum(0);
    for (size_t i = 0; i < _iterations; ++i)
    {
        sum += measureCall(device, elf, statement);
    }
    _measurements.push_back(sum / _iterations);
}

void BenchmarkRunner::runBenchmark(const ProblemStatement& statement, const ElfFactory& factory)
{
    std::unique_ptr<Elf> elf = factory.createElf(statement.elfCategory);
    if (_rank == MASTER)
    {
        for (int device = 0; device < _devices; ++device)
        {
            benchmarkDevice(device, *elf, statement);
        }
    }
    else
    {
        benchmarkDevice(MASTER, *elf, statement);
        _measurements.clear();
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
