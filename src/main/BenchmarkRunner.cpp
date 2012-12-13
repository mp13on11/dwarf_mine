
#include <mpi.h>
#include <vector>
#include "../elves/Elf.h"
#include "BenchmarkRunner.h"

using namespace std;

const int MASTER = 0;

BenchmarkRunner::BenchmarkRunner(size_t iterations)
    : _iterations(iterations), _rank(MPI::COMM_WORLD.Get_rank()), 
      _devices(MPI::COMM_WORLD.Get_size())
{
    
}

chrono::microseconds BenchmarkRunner::measureCall(int device, Elf* elf, const ProblemStatement& statement)
{
    typedef chrono::high_resolution_clock clock;
    clock::time_point before = clock::now();
    if (_rank == MASTER)
    {
        if (device == MASTER)
        {
            elf->setInput(statement.input);
            elf->process();
            auto outstream = elf->getOutput();
        }
        else
        {
            vector<char> buffer;
            while (statement.input.good())
            {
                char c;
                statement.input.get(&c)
                if (statement.input.gcount() == 0)
                {
                    break;
                }
                buffer.push_back(c);
            }
            statement.input.seekg(0);
            MPI::COMM_WORLD.Send(
                const_cast<char>(buffer), buffer.size(), MPI::CHAR,
                device, 0, MPI::COMM_WORLD);

            MPI::Status status;
            MPI::COMM_WORLD.Probe(device, 0, MPI::COMM_WORLD, &status);
            int bufferSize = status.Get_count(MPI::CHAR);
            char* buffer = new char[bufferSize];
            MPI::COMM_WORLD.Recv(buffer, bufferSize, MPI::CHAR, device, 0);
            delete[] buffer;
        }
    }
    else
    {
        MPI::Status status;
        MPI::COMM_WORLD.Probe(MASTER, 0, MPI::COMM_WORLD, &status);
        int bufferSize = status.Get_count(MPI::CHAR);
        char* buffer = new char[bufferSize];
        MPI::COMM_WORLD.Recv(buffer, bufferSize, MPI::CHAR, device, 0);
        MPI::Status status;
        MPI::COMM_WORLD.Probe(device, 0, MPI::COMM_WORLD, &status);
        int bufferSize = status.Get_count(MPI::CHAR);
        char* buffer = new char[bufferSize];
        MPI::COMM_WORLD.Recv(buffer, bufferSize, MPI::CHAR, device, 0);
        elf->setInput(buffer);
        elf->process();
        auto stream = elf->GetOutput();
        MPI::COMM_WORLD.Send();
        delete buffer[];
    }
    return clock::now() - before;
}

void BenchmarkRunner::runBenchmark(const ProblemStatement& statement)
{
    Elf* elf = ElfFactory.Create(statement.elfCategory);
    for (int device = 0; device < _devices; ++device)
    {
        chrono::microseconds sum = chrono::microseconds(0);
        for (size_t i = 0; i < _iterations; ++i)
        {
            sum += measureCall(device, statement);
        }
        _measurements.push_back(sum / _iterations);
    }
    delete elf;
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