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
    cout << "DISPATCH!" << endl;
    if (MpiHelper::isMaster())
        orchestrateCalculation();
    else
        calculateOnSlave();
}

void MatrixOnlineScheduler::orchestrateCalculation()
{
    cout << "ORCH" << endl;
    sliceInput();
    cout << "SDKLGFNSDKLFG" << endl; 
    distributeToSlaves();
    collectResults(/*SLICES, */result);
}

void MatrixOnlineScheduler::sliceInput()
{
    MatrixSlicerOnline slicer;
    sliceDefinitions = slicer.layout(left.rows(), right.columns(), 4, 4);
    currentSliceDefinition = sliceDefinitions.cbegin();
}

void MatrixOnlineScheduler::distributeToSlaves()
{
    while (hasSlices() || !haveSlavesFinished())
    {
        MatrixPair requestedSlices;
        cout << "Awaiting request" << endl;
        NodeId requestingNode = MatrixHelper::getNextSliceRequest();
        cout << "Request from slave " << requestingNode << endl;
        if (hasSlices())
        {
            requestedSlices = sliceMatrices(*currentSliceDefinition);
            currentSliceDefinition++;
        }
        else
        {
            requestedSlices = MatrixPair(Matrix<float>(), Matrix<float>());
            finishedWorkers[requestingNode-1] = true;
        }
        MatrixHelper::sendMatrixTo(requestedSlices.first, requestingNode);
        MatrixHelper::sendMatrixTo(requestedSlices.second, requestingNode);
    }
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
    cout << "Slave " << MpiHelper::rank() << " requesting slice." << endl;
    MatrixHelper::requestNextSlice(MpiHelper::rank());
    cout << "Slave " << MpiHelper::rank() << " fetching left slice." << endl;
    Matrix<float> left = MatrixHelper::receiveMatrixFrom(MpiHelper::MASTER);
    cout << "Slave " << MpiHelper::rank() << " fetching right slice." << endl;
    Matrix<float> right = MatrixHelper::receiveMatrixFrom(MpiHelper::MASTER);
    while (left.rows() > 0 && left.columns() > 0 && right.rows() > 0 && right.columns() > 0)
    {
    cout << "Slave " << MpiHelper::rank() << " requesting slice." << endl;
        MatrixHelper::requestNextSlice(MpiHelper::rank());
    cout << "Slave " << MpiHelper::rank() << " fetching left slice." << endl;
        left = MatrixHelper::receiveMatrixFrom(MpiHelper::MASTER);
    cout << "Slave " << MpiHelper::rank() << " fetching right slice." << endl;
        right = MatrixHelper::receiveMatrixFrom(MpiHelper::MASTER);
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

