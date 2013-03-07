#include "Elf.h"
#include "Matrix.h"
#include "MatrixElf.h"
#include "MatrixHelper.h"
#include "MatrixOnlineScheduler.h"
#include "MatrixSlice.h"
#include "MatrixSlicer.h"
#include "MatrixSlicerSquarified.h"
#include "common/ProblemStatement.h"

#include <iostream>
#include <memory>
#include <sstream>
#include <random>

using namespace std;
using MatrixHelper::MatrixPair;

MatrixOnlineScheduler::MatrixOnlineScheduler(const function<ElfPointer()>& factory) :
    MatrixScheduler(factory)
{
}

MatrixOnlineScheduler::~MatrixOnlineScheduler()
{
}

void MatrixOnlineScheduler::doSimpleDispatch()
{
    result = elf().multiply(left, right);
}

void MatrixOnlineScheduler::doDispatch()
{
    if (MpiHelper::isMaster())
    {
        orchestrateCalculation();
    }
    else
    {
        calculateOnSlave();
    }
}

void MatrixOnlineScheduler::orchestrateCalculation()
{
    result = dispatchAndReceive();
}

Matrix<float> MatrixOnlineScheduler::dispatchAndReceive() const
{
    Matrix<float> result(left.rows(), right.columns());
    MatrixSlicerSquarified slicer;
    vector<MatrixSlice> sliceDefinitions = slicer.layout(nodeSet, result.rows(), result.columns());
    const MatrixSlice* masterSlice = nullptr; //distributeSlices(sliceDefinitions);
    if (masterSlice != nullptr)
    {
        calculateOnMaster(*masterSlice, result);
    }
    collectResults(sliceDefinitions, result);
    return result;
}

void MatrixOnlineScheduler::calculateOnSlave()
{
    Matrix<float> left = MatrixHelper::receiveMatrixFrom(MpiHelper::MASTER);
    Matrix<float> right = MatrixHelper::receiveMatrixFrom(MpiHelper::MASTER);
    Matrix<float> result = elf().multiply(left, right);
    MatrixHelper::sendMatrixTo(result, MpiHelper::MASTER);
}

void MatrixOnlineScheduler::calculateOnMaster(const MatrixSlice& sliceDefinition, Matrix<float>& result) const
{
    MatrixPair slicedMatrices = sliceMatrices(sliceDefinition);
    Matrix<float> resultSlice = elf().multiply(slicedMatrices.first, slicedMatrices.second);
    sliceDefinition.injectSlice(resultSlice, result);
}

void MatrixOnlineScheduler::collectResults(const vector<MatrixSlice>& sliceDefinitions, Matrix<float>& result) const
{
    for (const MatrixSlice& definition : sliceDefinitions)
    {
        auto nodeId = definition.getNodeId();
        if (!MpiHelper::isMaster(nodeId))
        {
            Matrix<float> resultSlice = MatrixHelper::receiveMatrixFrom(nodeId);
            definition.injectSlice(resultSlice, result);
        }
    }
}

