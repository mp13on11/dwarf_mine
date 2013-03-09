#include "Elf.h"
#include "Matrix.h"
#include "MatrixElf.h"
#include "MatrixHelper.h"
#include "MatrixOnlineScheduler.h"
#include "MatrixSlice.h"
#include "MatrixSlicer.h"
#include "MatrixSlicerOnline.h"
#include "common/ProblemStatement.h"

#include <future>
#include <iostream>

using namespace std;
using MatrixHelper::MatrixPair;

bool MatrixOnlineScheduler::finishedWorkers[4] = {false, false, false, false};

MatrixOnlineScheduler::MatrixOnlineScheduler(const function<ElfPointer()>& factory) :
    MatrixScheduler(factory)
{
}

MatrixOnlineScheduler::~MatrixOnlineScheduler()
{
}

void MatrixOnlineScheduler::doDispatch()
{
    if (MpiHelper::isMaster())
        orchestrateCalculation();
    else
        calculateOnSlave();
}

void MatrixOnlineScheduler::orchestrateCalculation()
{
    sliceInput();
    schedule();
}

void MatrixOnlineScheduler::sliceInput()
{
    MatrixSlicerOnline slicer;
    result = Matrix<float>(left.rows(), right.columns());
    sliceDefinitions = slicer.layout(result.rows(), result.columns(), 4, 4);
    currentSliceDefinition = sliceDefinitions.begin();
}

void MatrixOnlineScheduler::schedule()
{
    while (hasSlices() || !haveSlavesFinished())
    {
        NodeId requestingNode;
        cout << "Awaiting request." << endl;
        requestingNode = MatrixHelper::getNextSliceRequest();
        cout << "Request from slave " << requestingNode << endl;
        fetchResultFrom(requestingNode);
        sendNextSlicesTo(requestingNode);
    }
}

void MatrixOnlineScheduler::fetchResultFrom(const NodeId node)
{
    cout << "Receiving slave " << node << "'s result." << endl;
    Matrix<float> nodeResult = MatrixHelper::receiveMatrixFrom(node);
    if (nodeResult.empty()) return;
    MatrixSlice& sliceDefinition = getNextSliceDefinitionFor(node);
    sliceDefinition.injectSlice(nodeResult, result);
    sliceDefinition.setNodeId(MpiHelper::MASTER);
}

MatrixSlice& MatrixOnlineScheduler::getNextSliceDefinitionFor(const NodeId node)
{
    for (auto& slice : sliceDefinitions)
        if (slice.getNodeId() == node)
            return slice;
    throw "ERROR: No next slice definition found.";
}

void MatrixOnlineScheduler::sendNextSlicesTo(const NodeId node)
{
    MatrixPair requestedSlices;
    if (hasSlices())
    {
        requestedSlices = sliceMatrices(*currentSliceDefinition);
        (*currentSliceDefinition).setNodeId(node);
        currentSliceDefinition++;
    }
    else
    {
        requestedSlices = MatrixPair(Matrix<float>(0, 0), Matrix<float>(0, 0));
        finishedWorkers[node-1] = true;
    }
    cout << "Sending slices to slave " << node << endl;
    MatrixHelper::sendMatrixTo(requestedSlices.first, node);
    MatrixHelper::sendMatrixTo(requestedSlices.second, node);
}

bool MatrixOnlineScheduler::hasSlices() const
{
    return currentSliceDefinition != sliceDefinitions.end();
}

bool MatrixOnlineScheduler::haveSlavesFinished() const
{
    return finishedWorkers[0]
        && finishedWorkers[1]
        && finishedWorkers[2]
        && finishedWorkers[3];
}

void MatrixOnlineScheduler::calculateOnSlave()
{
    Matrix<float> left, right, result = Matrix<float>(0, 0);
    calculateNextResult(result, left, right);
    while (!left.empty() && !right.empty())
        calculateNextResult(result, left, right);
}

void MatrixOnlineScheduler::calculateNextResult(
    Matrix<float>& result,
    Matrix<float>& left,
    Matrix<float>& right)
{
    int rank = MpiHelper::rank();
    MatrixHelper::requestNextSlice(rank);
    MatrixHelper::sendMatrixTo(result, MpiHelper::MASTER);
    left = MatrixHelper::receiveMatrixFrom(MpiHelper::MASTER);
    right = MatrixHelper::receiveMatrixFrom(MpiHelper::MASTER);
    result = elf().multiply(left, right);
}

