
#include <mpi.h>
#include <vector>
#include <sstream>
#include "Elf.h"
#include "BenchmarkRunner.h"

using namespace std;

const int MASTER = 0;

BenchmarkRunner::BenchmarkRunner(size_t iterations)
    : _iterations(iterations), _rank(MPI::COMM_WORLD.Get_rank()), 
      _devices(MPI::COMM_WORLD.Get_size())
{
    
}

void receive(ostream& out, int fromRank, int _rank)
{
    MPI::Status status;
    MPI::COMM_WORLD.Probe(fromRank, 0, status);
    int bufferSize = status.Get_count(MPI::CHAR);
    char* bufferA = new char[bufferSize];
    MPI::COMM_WORLD.Recv(bufferA, bufferSize, MPI::CHAR, fromRank, 0);
    out.write(bufferA, bufferSize);
    delete[] bufferA;
}

void send(istream& in, int toRank, int _rank)
{
    vector<char> buffer;
    while (in.good())
    {
        char c;
        in.get(c);
        if (in.gcount() == 0)
        {
            break;
        }
        buffer.push_back(c);
    }
    MPI::COMM_WORLD.Send(const_cast<char*>(buffer.data()), buffer.size(), MPI::CHAR, toRank, 0);
}

chrono::microseconds BenchmarkRunner::measureCall(int targetRank, Elf& elf, const ProblemStatement& statement)
{
    typedef chrono::high_resolution_clock clock;
    clock::time_point before = clock::now();
    if (_rank == MASTER)
    {
        statement.input.seekg (0, ios::beg);
        if (targetRank == MASTER)
        {
            stringstream out;
            elf.run(statement.input, out);
        }
        else
        {
            send(statement.input, targetRank, _rank);
            stringstream output;
            receive(output, targetRank, _rank);
        }
    }
    else
    {
        stringstream input;
        stringstream output;
        receive(input, MASTER, _rank);
        elf.run(input, output);
        send(output, MASTER, _rank);
    }
    return clock::now() - before;
}

void BenchmarkRunner::runBenchmark(const ProblemStatement& statement, const ElfFactory& factory)
{
    std::unique_ptr<Elf> elf = factory.createElf(statement.elfCategory);
    if (_rank == MASTER)
    {
        for (int device = 0; device < _devices; ++device)
        {
            chrono::microseconds sum = chrono::microseconds(0);
            for (size_t i = 0; i < _iterations; ++i)
            {
                sum += measureCall(device, *elf, statement);
            }
            _measurements.push_back(sum / _iterations);
        }
    }
    else
    {
        for (size_t i = 0; i < _iterations; ++i)
        {
            measureCall(MASTER, *elf, statement);
        }
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