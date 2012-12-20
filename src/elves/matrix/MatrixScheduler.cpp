#include "MatrixScheduler.h"
#include <Elf.h>
#include <main/ProblemStatement.h>
#include <sstream>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <numeric>
#include "Matrix.h"
#include <cmath>
#include <list>
#include <map>

using namespace std;

const int MASTER = 0; // TODO: Put in one common .h file

struct MatrixSlice
{
    size_t x;
    size_t y;
    size_t columns;
    size_t rows;
};

void spliceColumns(vector<MatrixSlice>& slices, list<int>& relation, int rowOrigin, int columnOrigin, int rows, int columns);
void spliceRows(vector<MatrixSlice>& slices, list<int>& relation, int rowOrigin, int columnOrigin, int rows, int columns);


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

void spliceColumns(vector<MatrixSlice>& slices, list<int>& relation, int rowOrigin, int columnOrigin, int rows, int columns)
{
    int processor = relation.size() - 1;
    if (processor < 0)
    {
        return;
    }

    int pivot = columns;
    if (relation.size() > 1)
    {
        int overall = 0;
        for (const auto& s : relation)
        {
            overall += s;
        }
        pivot = ceil(columns * relation.front() * 1.0 / overall);
    }
    slices[processor].x = columnOrigin;
    slices[processor].y = rowOrigin;
    slices[processor].columns = columnOrigin;
    slices[processor].rows = rows + pivot;
    relation.pop_front();
    spliceRows(slices, relation, rowOrigin, columnOrigin + pivot, rows, columns - pivot);
}

void spliceRows(vector<MatrixSlice>& slices, list<int>& relation, int rowOrigin, int columnOrigin, int rows, int columns)
{
    int processor = relation.size() - 1;
    if (processor < 0)
    {
        cout << relation.size() << endl;
        return;
    }

    int pivot = rows;
    if (relation.size() > 1)
    {
        int overall = 0;
        for (const auto& s : relation)
        {
            overall += s;
        }
        pivot = ceil(rows * relation.front() * 1.0 / overall);
    }
    slices[processor].x = columnOrigin;
    slices[processor].y = rowOrigin;
    slices[processor].columns = columnOrigin + pivot;
    slices[processor].rows = rows;
    relation.pop_front();
    spliceColumns(slices, relation, rowOrigin + pivot, columnOrigin, rows - pivot, columns);
}

vector<MatrixSlice> spliceAndDice(Matrix<float>& map, list<int>& relation)
{
    vector<MatrixSlice> slices(relation.size());
    spliceColumns(slices, relation, 0, 0, map.rows(), map.columns());
    return slices;
}

// vector<char> packMatrices(const vector<Matrix<float>>& matrices)
// {

// }

// vector<Matrix<float>> unpackMatrices(const vector<char> buffer)
// {

// }


void MatrixScheduler::doDispatch(ProblemStatement& statement)
{
    // Matrix<float> _map(93,37);
    // list<int> relation ={5, 3, 3, 3, 1};
    // spliceAndDice(_map, relation);
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
