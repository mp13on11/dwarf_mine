#include "Elf.h"
#include "Matrix.h"
#include "MatrixElf.h"
#include "MatrixHelper.h"
#include "MatrixScheduler.h"
#include "MatrixSlice.h"
#include "MatrixSlicer.h"
#include "main/ProblemStatement.h"

#include <iostream>
#include <memory>
#include <sstream>

using namespace std;
using MatrixHelper::MatrixPair;

struct MatrixScheduler::MatrixSchedulerImpl
{
    MatrixSchedulerImpl(MatrixScheduler* self) : self(self) {}

    void orchestrateCalculation();
    void calculateOnSlave();
    Matrix<float> dispatchAndReceive(const MatrixPair& matrices);
    const MatrixSlice* distributeSlices(const std::vector<MatrixSlice>& sliceDefinitions, const MatrixPair& matrices);
    void calculateOnMaster(const MatrixSlice& sliceDefinition, const MatrixPair& matrices, Matrix<float>& result);
    void collectResults(const std::vector<MatrixSlice>& sliceDefinitions, Matrix<float>& result);

    void outputData(ProblemStatement& statement);
    bool hasData();
    void provideData(ProblemStatement& statement);

    // Reference to containing MatrixScheduler
    MatrixScheduler* self;
    MatrixPair matrices;
    Matrix<float> result;
};

MatrixScheduler::MatrixScheduler(const function<ElfPointer()>& factory) :
    SchedulerTemplate(factory), pImpl(new MatrixSchedulerImpl(this))
{
}

MatrixScheduler::~MatrixScheduler()
{
    delete pImpl;
}

void MatrixScheduler::provideData(ProblemStatement& statement)
{
	cout << "providing data..." << endl;
    pImpl->provideData(statement);
}

bool MatrixScheduler::hasData()
{
    return pImpl->hasData();
}

void MatrixScheduler::outputData(ProblemStatement& statement)
{
    pImpl->outputData(statement);
}

void MatrixScheduler::doDispatch()
{
    if (MpiHelper::isMaster(rank))
    {
        pImpl->orchestrateCalculation();
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

    return { move(slicedLeft), move(slicedRight) };
}

void MatrixScheduler::MatrixSchedulerImpl::calculateOnSlave()
{
    Matrix<float> left = MatrixHelper::receiveMatrixFrom(MpiHelper::MASTER);
    Matrix<float> right = MatrixHelper::receiveMatrixFrom(MpiHelper::MASTER);
    Matrix<float> result = self->elf->multiply(left, right);
    MatrixHelper::sendMatrixTo(result, MpiHelper::MASTER);
}

void MatrixScheduler::MatrixSchedulerImpl::provideData(ProblemStatement& statement)
{
    statement.input->clear();
    statement.input->seekg(0);
    matrices = MatrixHelper::readMatrixPairFrom(*(statement.input));
}

void MatrixScheduler::MatrixSchedulerImpl::outputData(ProblemStatement& statement)
{
    MatrixHelper::writeMatrixTo(*(statement.output), result);
}

bool MatrixScheduler::MatrixSchedulerImpl::hasData()
{
    return !matrices.first.empty() || !matrices.second.empty();
}

void MatrixScheduler::MatrixSchedulerImpl::orchestrateCalculation()
{
    result = dispatchAndReceive(matrices);
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
        if (MpiHelper::isMaster(nodeId))
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
        if (!MpiHelper::isMaster(nodeId))
        {
            Matrix<float> resultSlice = MatrixHelper::receiveMatrixFrom(nodeId);
            definition.injectSlice(resultSlice, result);
        }
    }
}

void MatrixScheduler::MatrixSchedulerImpl::calculateOnMaster(const MatrixSlice& sliceDefinition, const MatrixPair& matrices, Matrix<float>& result)
{
    MatrixPair slicedMatrices = sliceMatrices(sliceDefinition, matrices);
    Matrix<float> resultSlice = self->elf->multiply(slicedMatrices.first, slicedMatrices.second);
    sliceDefinition.injectSlice(resultSlice, result);
}
