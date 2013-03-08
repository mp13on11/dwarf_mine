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
    collectResults();
}

void MatrixOnlineScheduler::sliceInput()
{
    MatrixSlicerOnline slicer;
    sliceDefinitions = slicer.layout(left.rows(), right.columns(), 4, 4);
    currentSliceDefinition = sliceDefinitions.cbegin();
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
    // TODO: Inject nodeResult into own result
}

void MatrixOnlineScheduler::sendNextSlicesTo(const NodeId node)
{
    MatrixPair requestedSlices;
    if (hasSlices())
    {
        requestedSlices = sliceMatrices(*currentSliceDefinition);
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
    return currentSliceDefinition != sliceDefinitions.cend();
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

void MatrixOnlineScheduler::collectResults()
{
    result = Matrix<float>();
    cout << "Collecting results... (not)" << endl;
}

