#include "MatrixScheduler.h"
#include "Matrix.h"
#include "MatrixHelper.h"
#include "MatrixSlice.h"
#include <Elf.h>
#include <main/ProblemStatement.h>
#include <sstream>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <numeric>
#include <cmath>
#include <list>
#include <map>

using namespace std;

void sliceColumns(vector<MatrixSlice>& slices, list<NodeRating>& ratings, size_t rowOrigin, size_t columnOrigin, size_t rows, size_t columns);
void sliceRows(vector<MatrixSlice>& slices, list<NodeRating>& ratings, size_t rowOrigin, size_t columnOrigin, size_t rows, size_t columns);

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

void sliceColumns(vector<MatrixSlice>& slices, list<NodeRating>& ratings, size_t rowOrigin, size_t columnOrigin, size_t rows, size_t columns)
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
    slices.push_back(MatrixSlice{processor, columnOrigin, rowOrigin, columnOrigin + pivot, rows});
    ratings.pop_front();
    sliceRows(slices, ratings, rowOrigin, columnOrigin + pivot, rows, columns - pivot);
}

void sliceRows(vector<MatrixSlice>& slices, list<NodeRating>& ratings, size_t rowOrigin, size_t columnOrigin, size_t rows, size_t columns)
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
    slices.push_back(MatrixSlice{processor, columnOrigin, rowOrigin, columns, rowOrigin + pivot});
    ratings.pop_front();
    sliceColumns(slices, ratings, rowOrigin + pivot, columnOrigin, rows - pivot, columns);
}

vector<MatrixSlice> sliceAndDice(BenchmarkResult& results, size_t rows, size_t columns)
{
    list<NodeRating> orderedRatings(results.begin(), results.end());
    orderedRatings.sort(
        [](const NodeRating& a, const NodeRating& b)
        {
            return a.second > b.second;
        }
    );

    vector<MatrixSlice> slicesDefinitions;
    sliceColumns(slicesDefinitions, orderedRatings, 0, 0, rows, columns);
    return slicesDefinitions;
}

pair<Matrix<float>, Matrix<float>> sliceMatrices(const MatrixSlice& definition, const pair<Matrix<float>, Matrix<float>>& matrices)
{
    Matrix<float> slicedLeft = definition.extractSlice(matrices.first, true);
    Matrix<float> slicedRight = definition.extractSlice(matrices.second, false);

    return make_pair<Matrix<float>, Matrix<float>>(move(slicedLeft), move(slicedRight));
}

MatrixScheduler::MatrixScheduler(const BenchmarkResult& benchmarkResult) :
    Scheduler(benchmarkResult)
{
}

void MatrixScheduler::doDispatch(ProblemStatement& statement)
{
    if (rank == MASTER)
    {
        statement.input.clear();
        statement.input.seekg(0);
        pair<Matrix<float>, Matrix<float>> matrices = MatrixHelper::readMatrixPairFrom(statement.input);
        Matrix<float> result(matrices.first.rows(), matrices.second.columns());

        vector<MatrixSlice> sliceDefinitions = sliceAndDice(nodeSet, result.rows(), result.columns());
        bool calculateOnMaster = false;
        const MatrixSlice* masterSliceDefinition = nullptr;

        for(const auto& definition : sliceDefinitions)
        {
            if (definition.getNodeId() == MASTER)
            {
                calculateOnMaster = true;
                masterSliceDefinition = &definition;
            }
            else
            {
                auto inputMatrices = sliceMatrices(definition, matrices);
                definition.send();
                //stringstream buffer;
                //MatrixHelper::writeMatrixPairTo(buffer, );
                //send(buffer, definition.getNodeId());
            }
        }
        if (calculateOnMaster)
        {
            pair<Matrix<float>, Matrix<float>> slicedMatrices = sliceMatrices(*masterSliceDefinition, matrices);

            stringstream inputBuffer;
            stringstream outputBuffer;
            MatrixHelper::writeMatrixPairTo(inputBuffer, sliceMatrices(*masterSliceDefinition, matrices));

            elf->run(inputBuffer, outputBuffer);

            Matrix<float> resultSlice = MatrixHelper::readMatrixFrom(outputBuffer);
            //Matrix<float> resultSlice = elf->run(slicedMatrices.first, slicedMatrices.second);

            masterSliceDefinition->injectSlice(resultSlice, result);
        }
        for (const auto& definition : sliceDefinitions)
        {
            if (definition.getNodeId() != MASTER)
            {
                stringstream buffer;
                receive(buffer, definition.getNodeId());
                Matrix<float> resultSlice = MatrixHelper::readMatrixFrom(buffer);

                definition.injectSlice(resultSlice, result);
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
