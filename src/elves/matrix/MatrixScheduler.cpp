#include "MatrixScheduler.h"
#include <Elf.h>
#include <main/ProblemStatement.h>
#include <sstream>
#include <iostream>
#include <memory>
#include <mpi.h>

using namespace std;

const int MASTER = 0; // TODO: Put in one common .h file

MatrixScheduler::MatrixScheduler(const BenchmarkResult& benchmarkResult) :
    Scheduler(benchmarkResult)
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

void MatrixScheduler::doDispatch(ProblemStatement& statement)
{
    if (rank == MASTER)
    {
        if (nodeSet.size() == 1)
        {
            int targetRank = nodeSet.begin()->first;

            statement.input.clear();
            statement.input.seekg(0, ios::beg);
            stringstream output;
            if (targetRank == MASTER)
            {
                elf->run(statement.input, output);
            }
            else
            {
                send(statement.input, targetRank);
                receive(output, targetRank);
            }
        }
    }
    else
    {
        receive(statement.input, MASTER);
        elf->run(statement.input, statement.output);
        send(statement.output, MASTER);
    }
}
