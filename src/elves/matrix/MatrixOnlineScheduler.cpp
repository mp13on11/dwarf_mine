#include "Elf.h"
#include "Matrix.h"
#include "MatrixElf.h"
#include "MatrixHelper.h"
#include "MatrixOnlineScheduler.h"
#include "MatrixSlice.h"
#include "MatrixSlicer.h"
#include "common/ProblemStatement.h"

#include <future>
#include <iostream>

using namespace std;
using MatrixHelper::MatrixPair;

int MatrixOnlineScheduler::slices = 100;
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
    distributeToSlaves();
    collectResults(/*SLICES, */result);
}

void MatrixOnlineScheduler::sliceInput()
{
    // Slicing
}

void MatrixOnlineScheduler::distributeToSlaves()
{
    while (hasSlices() || !haveSlavesFinished())
    {
        Matrix<float> requestedSlice;
        NodeId requestingNode = MatrixHelper::getNextSliceRequest();
        if (hasSlices())
        {
            requestedSlice = Matrix<float>(1, 1);
            slices--;
        }
        else
        {
            requestedSlice = Matrix<float>(0, 0);
            finishedWorkers[requestingNode-1] = true;
        }
        MatrixHelper::sendMatrixTo(requestedSlice, requestingNode);
    }
}

bool MatrixOnlineScheduler::hasSlices() const
{
    return slices > 0;
}

bool MatrixOnlineScheduler::haveSlavesFinished() const
{
    return finishedWorkers[0];
        //&& finishedWorkers[1]
        //&& finishedWorkers[2];
        //&& finishedWorkers[3];
}

void MatrixOnlineScheduler::calculateOnSlave()
{
    MatrixHelper::requestNextSlice(MpiHelper::rank());
    Matrix<float> slice = MatrixHelper::receiveMatrixFrom(MpiHelper::MASTER);
    while (slice.rows() == 1 && slice.columns() == 1)
    {
        MatrixHelper::requestNextSlice(MpiHelper::rank());
        slice = MatrixHelper::receiveMatrixFrom(MpiHelper::MASTER);
    }
    /*
    Matrix<float> left = MatrixHelper::receiveMatrixFrom(MpiHelper::MASTER);
    Matrix<float> right = MatrixHelper::receiveMatrixFrom(MpiHelper::MASTER);
    Matrix<float> result = elf().multiply(left, right);
    MatrixHelper::sendMatrixTo(result, MpiHelper::MASTER);
    */
}

void MatrixOnlineScheduler::collectResults(/*const vector<MatrixSlice>& sliceDefinitions, */Matrix<float>& result) const
{
    result = Matrix<float>();
    cout << "Collecting results... (not)" << endl;
    /*
    for (const MatrixSlice& definition : sliceDefinitions)
    {
        auto nodeId = definition.getNodeId();
        if (!MpiHelper::isMaster(nodeId))
        {
            Matrix<float> resultSlice = MatrixHelper::receiveMatrixFrom(nodeId);
            definition.injectSlice(resultSlice, result);
        }
    }
    */
}

