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

vector<MatrixSlice> MatrixOnlineScheduler::sliceDefinitions = std::vector<MatrixSlice>();
vector<MatrixSlice>::iterator MatrixOnlineScheduler::currentSliceDefinition = MatrixOnlineScheduler::sliceDefinitions.begin();
map<NodeId, bool> MatrixOnlineScheduler::finishedSlaves = map<NodeId, bool>();
vector<future<void>> MatrixOnlineScheduler::scheduleHandlers = vector<future<void>>();
mutex MatrixOnlineScheduler::schedulingMutex;

MatrixOnlineScheduler::MatrixOnlineScheduler(const function<ElfPointer()>& factory) :
    MatrixScheduler(factory)
{
    if (MpiHelper::isMaster())
        for (size_t i = 1; i < MpiHelper::numberOfNodes(); ++i)
            finishedSlaves[NodeId(i)] = false;
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
        const NodeId requestingNode = MatrixHelper::waitForSlicesRequest();
        scheduleHandlers.push_back(async(launch::async,
            [&, requestingNode] () { schedule(requestingNode); }));
    }
}

void MatrixOnlineScheduler::schedule(const NodeId node)
{
    const int lastWorkAmount = getLastWorkAmountFor(node);
    fetchResultsFrom(node, lastWorkAmount);
    sendNextSlicesTo(node);
}

void MatrixOnlineScheduler::fetchResultsFrom(const NodeId node, const int workAmount)
{
    for (int i = 0; i < workAmount; ++i)
    {
        Matrix<float> nodeResult = MatrixHelper::receiveMatrixFrom(node);
        if (nodeResult.empty()) return;
        MatrixSlice& sliceDefinition = getNextSliceDefinitionFor(node);
        sliceDefinition.injectSlice(nodeResult, result);
        sliceDefinition.setNodeId(MpiHelper::MASTER);
    }
}

int MatrixOnlineScheduler::getLastWorkAmountFor(const NodeId node) const
{
    return schedulingStrategy->getLastWorkAmountFor(*this, node);
}

int MatrixOnlineScheduler::getNextWorkAmountFor(const NodeId node) const
{
    return schedulingStrategy->getNextWorkAmountFor(*this, node);
}

vector<MatrixPair> MatrixOnlineScheduler::getNextWorkFor(
    const NodeId node,
    vector<MatrixSlice>::iterator& workSlice,
    const int workAmount)
{
    vector<MatrixPair> work;
    for (int i = 0; workSlice != sliceDefinitions.end() && i < workAmount; ++workSlice, ++i)
    {
        work.push_back(sliceMatrices(*workSlice));
        workSlice->setNodeId(node);
    }
    if ((int) work.size() < workAmount)
        work.push_back(MatrixPair(Matrix<float>(0,0), Matrix<float>(0,0)));
    return work;
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
    vector<MatrixSlice>::iterator workStartingSliceDefinition;
    schedulingMutex.lock();
    int workAmount = getNextWorkAmountFor(node);
    workStartingSliceDefinition = currentSliceDefinition;
    if (workAmount > getRemainingWorkAmount())
        currentSliceDefinition += getRemainingWorkAmount();
    else
        currentSliceDefinition += workAmount;
    schedulingMutex.unlock();
    const vector<MatrixPair> work = getNextWorkFor(node, workStartingSliceDefinition, workAmount);

    MatrixHelper::sendWorkAmountTo(node, workAmount);
    for (const auto& workPair : work)
    {
        if (workPair.first.empty() || workPair.second.empty())
            finishedSlaves[node] = true;
        MatrixHelper::sendMatrixTo(workPair.first, node);
        MatrixHelper::sendMatrixTo(workPair.second, node);
    }
}

int MatrixOnlineScheduler::getRemainingWorkAmount() const
{
    return hasSlices() ?
        distance(currentSliceDefinition, sliceDefinitions.end()) :
        0;
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
    MatrixHelper::requestNextSlices(MpiHelper::rank());
}

void MatrixOnlineScheduler::sendResults()
{
    for (const auto& result : resultQueue)
        MatrixHelper::sendMatrixTo(result, MpiHelper::MASTER);
}

void MatrixOnlineScheduler::receiveWork()
{
    const int workAmount = MatrixHelper::receiveWorkAmountFrom(MpiHelper::MASTER);
    for (int i = 0; i < workAmount; ++i)
    {
        Matrix<float> left = MatrixHelper::receiveMatrixFrom(MpiHelper::MASTER);
        Matrix<float> right = MatrixHelper::receiveMatrixFrom(MpiHelper::MASTER);
        workQueue.push_back({left, right});
    }
}

void MatrixOnlineScheduler::doWork()
{
    for (const auto& work : workQueue)
        resultQueue.push_back(elf().multiply(work.first, work.second));
}

