#include "MatrixScheduler.h"
#include "Matrix.h"
#include "MatrixHelper.h"
#include "MatrixSlice.h"
#include "MatrixSlicer.h"
#include "MatrixElf.h"
#include <Elf.h>
#include <main/ProblemStatement.h>
#include <sstream>
#include <iostream>
#include <memory>
#include <mpi.h>

using namespace std;
using MatrixHelper::MatrixPair;

struct MatrixScheduler::MatrixSchedulerImpl
{
    MatrixSchedulerImpl(MatrixScheduler* self) : self(self) {}

    void orchestrateCalculation(ProblemStatement& statement);
    void calculateOnSlave();
    Matrix<float> dispatchAndReceive(const MatrixPair& matrices);
    const MatrixSlice* distributeSlices(const std::vector<MatrixSlice>& sliceDefinitions, const MatrixPair& matrices);
    void calculateOnMaster(const MatrixSlice& sliceDefinition, const MatrixPair& matrices, Matrix<float>& result);
    void collectResults(const std::vector<MatrixSlice>& sliceDefinitions, Matrix<float>& result);

    // Reference to containing MatrixScheduler
    MatrixScheduler* self;
};

MatrixScheduler::MatrixScheduler() :
    pImpl(new MatrixSchedulerImpl(this))
{
}

MatrixScheduler::MatrixScheduler(const BenchmarkResult& benchmarkResult) :
    Scheduler(benchmarkResult), pImpl(new MatrixSchedulerImpl(this))
{
}

MatrixScheduler::~MatrixScheduler()
{
    delete pImpl;
}

void MatrixScheduler::doDispatch(ProblemStatement& statement)
{
    if (rank == MASTER)
    {
        pImpl->orchestrateCalculation(statement);
    }
    else
    {
        pImpl->calculateOnSlave();
    }
}

MatrixPair sliceMatrices(const MatrixSlice& definition, const MatrixPair& matrices)
{
    Matrix<float> slicedLeft = definition.extractSlice(matrices.first, true);
    Matrix<float> slicedRight = definition.extractSlice(matrices.second, false);

    return make_pair<Matrix<float>, Matrix<float>>(move(slicedLeft), move(slicedRight));
}

void MatrixScheduler::MatrixSchedulerImpl::calculateOnSlave()
{
    MatrixElf* elf = static_cast<MatrixElf*>(self->elf);
    Matrix<float> left = MatrixHelper::receiveMatrixFrom(MASTER);
    Matrix<float> right = MatrixHelper::receiveMatrixFrom(MASTER);
    Matrix<float> result = elf->multiply(left, right);
    MatrixHelper::sendMatrixTo(result, MASTER);
}

void MatrixScheduler::MatrixSchedulerImpl::orchestrateCalculation(ProblemStatement& statement)
{
    statement.input->clear();
    statement.input->seekg(0);
    MatrixPair matrices = MatrixHelper::readMatrixPairFrom(*(statement.input));
    Matrix<float> result = dispatchAndReceive(matrices);
    MatrixHelper::writeMatrixTo(*(statement.output), result);
}

Matrix<float> MatrixScheduler::MatrixSchedulerImpl::dispatchAndReceive(const MatrixPair& matrices)
{
    Matrix<float> result(matrices.first.rows(), matrices.second.columns());
    MatrixSlicer slicer;
    vector<MatrixSlice> sliceDefinitions = slicer.sliceAndDice(self->nodeSet, result.rows(), result.columns());
    const MatrixSlice* masterSlice = distributeSlices(sliceDefinitions, matrices);
    if (masterSlice != nullptr)
    {
        calculateOnMaster(*masterSlice, matrices, result);
    }
    collectResults(sliceDefinitions, result);
    return result;
}

const MatrixSlice* MatrixScheduler::MatrixSchedulerImpl::distributeSlices(const vector<MatrixSlice>& sliceDefinitions, const MatrixPair& matrices)
{
    const MatrixSlice* masterSliceDefinition = nullptr;

    for(const auto& definition : sliceDefinitions)
    {
        auto nodeId = definition.getNodeId();
        if (nodeId == MASTER)
        {
            masterSliceDefinition = &definition;
        }
        else
        {
            auto inputMatrices = sliceMatrices(definition, matrices);
            MatrixHelper::sendMatrixTo(inputMatrices.first, nodeId);
            MatrixHelper::sendMatrixTo(inputMatrices.second, nodeId);
        }
    }
    return masterSliceDefinition;
}

void MatrixScheduler::MatrixSchedulerImpl::collectResults(const vector<MatrixSlice>& sliceDefinitions, Matrix<float>& result)
{
    for (const auto& definition : sliceDefinitions)
    {
        auto nodeId = definition.getNodeId();
        if (nodeId != MASTER)
        {
            Matrix<float> resultSlice = MatrixHelper::receiveMatrixFrom(nodeId);
            definition.injectSlice(resultSlice, result);
        }
    }
}

void MatrixScheduler::MatrixSchedulerImpl::calculateOnMaster(const MatrixSlice& sliceDefinition, const MatrixPair& matrices, Matrix<float>& result)
{
    MatrixElf* elf = static_cast<MatrixElf*>(self->elf);
    MatrixPair slicedMatrices = sliceMatrices(sliceDefinition, matrices);
    Matrix<float> resultSlice = elf->multiply(slicedMatrices.first, slicedMatrices.second);
    sliceDefinition.injectSlice(resultSlice, result);
}
