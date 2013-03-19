#include "Elf.h"
#include "Matrix.h"
#include "MatrixElf.h"
#include "MatrixOnlineScheduler.h"
#include "MatrixSlice.h"
#include "MatrixSlicer.h"
#include "MatrixSlicerOnline.h"
#include "MatrixOnlineSchedulingStrategyFactory.h"
#include "common/ProblemStatement.h"

#include <algorithm>
#include <iterator>

using namespace std;
using MatrixHelper::MatrixPair;

std::vector<MatrixSlice> MatrixOnlineScheduler::sliceDefinitions = std::vector<MatrixSlice>();
std::vector<MatrixSlice>::iterator MatrixOnlineScheduler::currentSliceDefinition = MatrixOnlineScheduler::sliceDefinitions.begin();
map<int, bool> MatrixOnlineScheduler::finishedSlaves = map<int, bool>();

MatrixOnlineScheduler::MatrixOnlineScheduler(const Communicator& communicator, const function<ElfPointer()>& factory) :
    MatrixScheduler(communicator, factory)
{
    if (communicator.isMaster())
        for (size_t i = 1; i < communicator.size(); ++i)
            finishedSlaves[int(i)] = false;
    else
        resultQueue.push_back(Matrix<float>(0, 0));
}

MatrixOnlineScheduler::~MatrixOnlineScheduler()
{
}

void MatrixOnlineScheduler::configureWith(const Configuration& config)
{
    schedulingStrategy = MatrixOnlineSchedulingStrategyFactory::getStrategy(config.schedulingStrategy());
}

void MatrixOnlineScheduler::generateData(const DataGenerationParameters& params)
{
    MatrixScheduler::generateData(params);
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
    sliceDefinitions = schedulingStrategy->getSliceDefinitions(result, nodeSet);
    currentSliceDefinition = sliceDefinitions.begin();
}

void MatrixOnlineScheduler::schedule()
{
    while (hasSlices() || !haveSlavesFinished())
    {
        const int requestingNode = MatrixHelper::waitForSlicesRequest(communicator);
        const int workAmount = getWorkAmountFor(requestingNode);
        fetchResultsFrom(requestingNode, workAmount);
        sendNextSlicesTo(requestingNode, workAmount);
    }
}

void MatrixOnlineScheduler::fetchResultsFrom(const int node, const int workAmount)
{
    for (int i = 0; i < workAmount; ++i)
    {
        Matrix<float> nodeResult = MatrixHelper::receiveMatrixFrom(communicator, node);
        if (nodeResult.empty()) return;
        MatrixSlice& sliceDefinition = getNextSliceDefinitionFor(node);
        sliceDefinition.injectSlice(nodeResult, result);
        sliceDefinition.setNodeId(Communicator::MASTER_RANK);
    }
}

int MatrixOnlineScheduler::getWorkAmountFor(const int node) const
{
    return schedulingStrategy->getWorkAmountFor(node);
}

MatrixSlice& MatrixOnlineScheduler::getNextSliceDefinitionFor(const int node)
{
    for (auto& slice : sliceDefinitions)
        if (slice.getNodeId() == node)
            return slice;
    throw "ERROR: No next slice definition found.";
}

void MatrixOnlineScheduler::sendNextSlicesTo(const int node, const int workAmount)
{
    sendWorkAmountTo(node, workAmount);
    for (int i = 0; i < workAmount; ++i)
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
            finishedSlaves[node] = true;
        }
        MatrixHelper::sendMatrixTo(communicator, requestedSlices.first, node);
        MatrixHelper::sendMatrixTo(communicator, requestedSlices.second, node);
        if (finishedSlaves[node]) 
            return;
    }
}

void MatrixOnlineScheduler::sendWorkAmountTo(const int node, const int workAmount)
{
    const int actualWorkAmount = min(getRemainingWorkAmount() + 1, workAmount);
    MatrixHelper::sendWorkAmountTo(communicator, node, actualWorkAmount);
}

int MatrixOnlineScheduler::getRemainingWorkAmount()
{
    return distance(currentSliceDefinition, sliceDefinitions.end());
}

bool MatrixOnlineScheduler::hasSlices() const
{
    return currentSliceDefinition != sliceDefinitions.end();
}

bool MatrixOnlineScheduler::haveSlavesFinished() const
{
    for (const auto& slaveState : finishedSlaves)
        if (!slaveState.second)
            return false;
    return true;
}

void MatrixOnlineScheduler::calculateOnSlave()
{
    while (hasToWork())
    {
        workQueue.clear();
        initiateCommunication();
        sendResults();
        resultQueue.clear();
        receiveWork();
        doWork();
    }
    workQueue.clear();
    initiateCommunication();
    sendResults();
    resultQueue.clear();
}

bool MatrixOnlineScheduler::hasToWork()
{
    return workQueue.empty()
        || (!workQueue.back().first.empty() && !workQueue.back().second.empty());
}

void MatrixOnlineScheduler::initiateCommunication() const
{
    MatrixHelper::requestNextSlices(communicator, communicator.rank());
}

void MatrixOnlineScheduler::sendResults()
{
    for (const auto& result : resultQueue)
        MatrixHelper::sendMatrixTo(communicator, result, Communicator::MASTER_RANK);
}

void MatrixOnlineScheduler::receiveWork()
{
    const int workAmount = MatrixHelper::receiveWorkAmountFrom(communicator, Communicator::MASTER_RANK);
    for (int i = 0; i < workAmount; ++i)
    {
        Matrix<float> left = MatrixHelper::receiveMatrixFrom(communicator, Communicator::MASTER_RANK);
        Matrix<float> right = MatrixHelper::receiveMatrixFrom(communicator, Communicator::MASTER_RANK);
        workQueue.push_back({left, right});
    }
}

void MatrixOnlineScheduler::doWork()
{
    for (const auto& work : workQueue)
        resultQueue.push_back(elf().multiply(work.first, work.second));
}

