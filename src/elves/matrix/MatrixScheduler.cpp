/*****************************************************************************
* Dwarf Mine - The 13-11 Benchmark
*
* Copyright (c) 2013 Bünger, Thomas; Kieschnick, Christian; Kusber,
* Michael; Lohse, Henning; Wuttke, Nikolai; Xylander, Oliver; Yao, Gary;
* Zimmermann, Florian
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*****************************************************************************/

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

MatrixScheduler::MatrixScheduler(const Communicator& communicator, const function<ElfPointer()>& factory) :
    SchedulerTemplate(communicator, factory)
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
    if (communicator.isMaster())
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
    Matrix<float> left = MatrixHelper::receiveMatrixFrom(communicator, Communicator::MASTER_RANK);
    Matrix<float> right = MatrixHelper::receiveMatrixFrom(communicator, Communicator::MASTER_RANK);
    Matrix<float> result = elf().multiply(left, right);
    MatrixHelper::sendMatrixTo(communicator, result, Communicator::MASTER_RANK);
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
    //vector<MatrixSlice> sliceDefinitions = slicer.sliceAndDice(communicator.nodeSet(), result.rows(), result.columns());
    vector<MatrixSlice> sliceDefinitions = slicer.layout(
            communicator.nodeSet(), result.rows(), result.columns()
        );
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

        if (nodeId == Communicator::MASTER_RANK)
        {
            masterSliceDefinition = &definition;
        }
        else
        {
            auto inputMatrices = sliceMatrices(definition);
            MatrixHelper::sendMatrixTo(communicator, inputMatrices.first, nodeId);
            MatrixHelper::sendMatrixTo(communicator, inputMatrices.second, nodeId);
        }
    }

    return masterSliceDefinition;
}

void MatrixScheduler::collectResults(const vector<MatrixSlice>& sliceDefinitions, Matrix<float>& result) const
{
    for (const MatrixSlice& definition : sliceDefinitions)
    {
        auto nodeId = definition.getNodeId();
        if (nodeId != Communicator::MASTER_RANK)
        {
            Matrix<float> resultSlice = MatrixHelper::receiveMatrixFrom(communicator, nodeId);
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
