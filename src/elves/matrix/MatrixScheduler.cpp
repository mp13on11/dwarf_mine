#include "MatrixScheduler.h"
#include <Elf.h>
#include <main/ProblemStatement.h>
#include <sstream>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <numeric>
#include "Matrix.h"
#include "MatrixHelper.h"
#include <cmath>
#include <list>
#include <map>

using namespace std;

const int MASTER = 0; // TODO: Put in one common .h file

struct MatrixSliceDefinition
{
    NodeId node;
    size_t x;
    size_t y;
    size_t columns;
    size_t rows;
};

typedef pair<NodeId, Rating> NodeRating;

void sliceColumns(vector<MatrixSliceDefinition>& slices, list<NodeRating>& ratings, int rowOrigin, int columnOrigin, int rows, int columns);
void sliceRows(vector<MatrixSliceDefinition>& slices, list<NodeRating>& ratings, int rowOrigin, int columnOrigin, int rows, int columns);


MatrixScheduler::MatrixScheduler(const BenchmarkResult& benchmarkResult) :
    Scheduler(benchmarkResult)
{
}

void receive(iostream& stream, int fromRank)
{
    MPI::Status status;
    MPI::COMM_WORLD.Probe(fromRank, 0, status);
    int bufferSize = status.Get_count(MPI::CHAR);
    unique_ptr<char[]> buffer(new char[bufferSize]);
    MPI::COMM_WORLD.Recv(buffer.get(), bufferSize, MPI::CHAR, fromRank, 0);
    stream.write(buffer.get(), bufferSize);
    stream.clear();
    stream.seekg(0);
}

void send(iostream& stream, int toRank)
{
    stringstream buffer;
    buffer << stream.rdbuf();
    auto buffered = buffer.str();
    MPI::COMM_WORLD.Send(buffered.c_str(), buffered.size(), MPI::CHAR, toRank, 0);
}

void sliceColumns(vector<MatrixSliceDefinition>& slices, list<NodeRating>& ratings, int rowOrigin, int columnOrigin, int rows, int columns)
{
    if (ratings.size() == 0)
    {
        return;
    }
    int processor = ratings.front().first;

    int pivot = columns;
    if (ratings.size() > 1)
    {
        int overall = 0;
        for (const auto& s : ratings)
        {
            overall += s.second;
        }
        pivot = ceil(columns * ratings.front().second * 1.0 / overall);
    }
    slices[processor].node = processor;
    slices[processor].x = columnOrigin;
    slices[processor].y = rowOrigin;
    slices[processor].columns = columnOrigin;
    slices[processor].rows = rows + pivot;
    ratings.pop_front();
    sliceRows(slices, ratings, rowOrigin, columnOrigin + pivot, rows, columns - pivot);
}

void sliceRows(vector<MatrixSliceDefinition>& slices, list<NodeRating>& ratings, int rowOrigin, int columnOrigin, int rows, int columns)
{
    if (ratings.size() == 0)
    {
        return;
    }
    int processor = ratings.front().first;
    
    int pivot = rows;
    if (ratings.size() > 1)
    {
        int overall = 0;
        for (const auto& s : ratings)
        {
            overall += s.second;
        }
        pivot = ceil(rows * ratings.front().second * 1.0 / overall);
    }
    slices[processor].node = processor;
    slices[processor].x = columnOrigin;
    slices[processor].y = rowOrigin;
    slices[processor].columns = columnOrigin + pivot;
    slices[processor].rows = rows;
    ratings.pop_front();
    sliceColumns(slices, ratings, rowOrigin + pivot, columnOrigin, rows - pivot, columns);
}

bool compareRatingsDesc(const NodeRating& a, const NodeRating& b)
{

    return a.second > b.second;
}

vector<MatrixSliceDefinition> sliceAndDice(BenchmarkResult& results, size_t rows, size_t columns)
{
    list<NodeRating> orderedRatings(results.begin(), results.end());
    orderedRatings.sort(compareRatingsDesc);

    vector<MatrixSliceDefinition> slicesDefinitions(orderedRatings.size());
    sliceColumns(slicesDefinitions, orderedRatings, 0, 0, rows, columns);
    return slicesDefinitions;
}

pair<Matrix<float>, Matrix<float>> sliceMatrices(const MatrixSliceDefinition& definition, pair<Matrix<float>, Matrix<float>> matrices)
{
    Matrix<float> slicedLeft(definition.rows, matrices.first.columns());
    Matrix<float> slicedRight(matrices.second.rows(), definition.columns);

    for(size_t row = 0; row <= slicedLeft.rows(); ++row)
    {
        for(size_t column = 0; column <= slicedLeft.columns(); ++column)
        {
            slicedLeft(row, column) += matrices.first(row + definition.y, column);
        }
    }

    for(size_t row = 0; row <= slicedRight.rows(); ++row)
    {
        for(size_t column = 0; column <= slicedRight.columns(); ++column)
        {
            slicedRight(row, column) += matrices.second(row, column + definition.x);
        }
    }

    return make_pair<Matrix<float>, Matrix<float>>(move(slicedLeft), move(slicedRight));
}

void injectSliceToResult(const MatrixSliceDefinition& definition, Matrix<float> resultSlice, Matrix<float>& result)
{
    for(size_t row = 0; row <= resultSlice.rows(); ++row)
    {
        for(size_t column = 0; column <= resultSlice.columns(); ++column)
        {
            result(row + definition.y, column + definition.x) = resultSlice(row, column);
        }
    }    
}

void MatrixScheduler::doDispatch(ProblemStatement& statement)
{
    if (rank == MASTER)
    {
        //statement.input.clear();
        //statement.input.seekg(0);
        pair<Matrix<float>, Matrix<float>> matrices = MatrixHelper::readMatrixPairFrom(statement.input);
        Matrix<float> result(matrices.first.rows(), matrices.second.columns());
        vector<MatrixSliceDefinition> sliceDefinitions = sliceAndDice(nodeSet, result.rows(), result.columns());
        MatrixSliceDefinition masterSliceDefinition;
        for(const auto& d :sliceDefinitions)
        {
            if (d.node == MASTER)
            {
                masterSliceDefinition = d;
            }
            else
            {
                stringstream buffer;
                MatrixHelper::writeMatrixPairTo(buffer, sliceMatrices(d, matrices));
                send(buffer, d.node);
            }
        }
        pair<Matrix<float>, Matrix<float>> slicedMatrices = sliceMatrices(masterSliceDefinition, matrices);
        stringstream inputBuffer;
        stringstream outputBuffer;
        MatrixHelper::writeMatrixPairTo(inputBuffer, sliceMatrices(masterSliceDefinition, matrices));
        elf->run(inputBuffer, outputBuffer);
        Matrix<float> resultSlice = MatrixHelper::readMatrixFrom(outputBuffer);
        //Matrix<float> resultSlice = elf->run(slicedMatrices.first, slicedMatrices.second);
        injectSliceToResult(masterSliceDefinition, resultSlice, result);

        for (const auto& d : sliceDefinitions)
        {
            if (d.node != MASTER)
            {
                stringstream buffer;
                receive(buffer, d.node);
                Matrix<float> resultSlice = MatrixHelper::readMatrixFrom(buffer);
                injectSliceToResult(d, resultSlice, result);
            }
        }
        MatrixHelper::writeMatrixTo(statement.output, result);
    }
    else
    {
        stringstream inputBuffer;
        receive(inputBuffer, MASTER);
        //pair<Matrix<float>, Matrix<float>> matrices = MatrixHelper::readMatrixPairFrom(inputBuffer);
        //Matrix<float> result = elf->run(matrices.first, matrices.second);
        stringstream outputBuffer;
        //MatrixHelper::writeMatrixTo(outputBuffer, result);
        elf->run(inputBuffer, outputBuffer);
        send(outputBuffer, MASTER);
    }
}
