#include "Elf.h"
#include "Matrix.h"
#include "MatrixElf.h"
#include "MatrixHelper.h"
#include "MatrixScheduler.h"
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

MatrixScheduler::MatrixScheduler(const function<ElfPointer()>& factory) :
    SchedulerTemplate(factory)
{
}

MatrixScheduler::~MatrixScheduler()
{
}

void MatrixScheduler::doSimpleDispatch()
{
    result = elf().multiply(left, right);
}

void MatrixScheduler::provideData(istream& input)
{
    input.clear();
    input.seekg(0);
    auto matrices = MatrixHelper::readMatrixPairFrom(input);
    left = matrices.first;
    right = matrices.second;
}

bool MatrixScheduler::hasData() const
{
    return !left.empty() || !right.empty();
}

void MatrixScheduler::outputData(ostream& output)
{
    MatrixHelper::writeMatrixTo(output, result);
}

void MatrixScheduler::generateData(const DataGenerationParameters& params)
{
    left  = Matrix<float>(params.leftRows, params.common);
    right = Matrix<float>(params.common, params.rightColumns);
    auto distribution = uniform_real_distribution<float> (-100, +100);
    auto engine = mt19937(time(nullptr));
    auto generator = bind(distribution, engine);
    MatrixHelper::fill(left, generator);
    MatrixHelper::fill(right, generator);
}

void MatrixScheduler::doDispatch()
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

void MatrixScheduler::calculateOnSlave()
{
    Matrix<float> left = MatrixHelper::receiveMatrixFrom(MpiHelper::MASTER);
    Matrix<float> right = MatrixHelper::receiveMatrixFrom(MpiHelper::MASTER);
    Matrix<float> result = elf().multiply(left, right);
    MatrixHelper::sendMatrixTo(result, MpiHelper::MASTER);
}

void MatrixScheduler::orchestrateCalculation()
{
    result = dispatchAndReceive();
}

Matrix<float> MatrixScheduler::dispatchAndReceive() const
{
    Matrix<float> result(left.rows(), right.columns());
    //MatrixSlicer slicer;
    MatrixSlicerSquarified slicer;
    //vector<MatrixSlice> sliceDefinitions = slicer.sliceAndDice(nodeSet, result.rows(), result.columns());
    vector<MatrixSlice> sliceDefinitions = slicer.layout(nodeSet, result.rows(), result.columns());
    const MatrixSlice* masterSlice = distributeSlices(sliceDefinitions);
    if (masterSlice != nullptr)
    {
        calculateOnMaster(*masterSlice, result);
    }
    collectResults(sliceDefinitions, result);
    return result;
}

const MatrixSlice* MatrixScheduler::distributeSlices(const vector<MatrixSlice>& sliceDefinitions) const
{
    const MatrixSlice* masterSliceDefinition = nullptr;

    for (const MatrixSlice& definition : sliceDefinitions)
    {
        auto nodeId = definition.getNodeId();

        if (MpiHelper::isMaster(nodeId))
        {
            masterSliceDefinition = &definition;
        }
        else
        {
            auto inputMatrices = sliceMatrices(definition);
            MatrixHelper::sendMatrixTo(inputMatrices.first, nodeId);
            MatrixHelper::sendMatrixTo(inputMatrices.second, nodeId);
        }
    }

    return masterSliceDefinition;
}

void MatrixScheduler::collectResults(const vector<MatrixSlice>& sliceDefinitions, Matrix<float>& result) const
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

void MatrixScheduler::calculateOnMaster(const MatrixSlice& sliceDefinition, Matrix<float>& result) const
{
    MatrixPair slicedMatrices = sliceMatrices(sliceDefinition);
    Matrix<float> resultSlice = elf().multiply(slicedMatrices.first, slicedMatrices.second);
    sliceDefinition.injectSlice(resultSlice, result);
}

MatrixPair MatrixScheduler::sliceMatrices(const MatrixSlice& definition) const
{
    Matrix<float> slicedLeft = definition.extractSlice(left, true);
    Matrix<float> slicedRight = definition.extractSlice(right, false);

    return { move(slicedLeft), move(slicedRight) };
}

void MatrixScheduler::doBenchmarkDispatch(NodeId /*node*/)
{
    dispatch();
}    

