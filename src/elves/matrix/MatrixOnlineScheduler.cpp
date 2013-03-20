#include "Elf.h"
#include "Matrix.h"
#include "MatrixElf.h"
#include "MatrixOnlineScheduler.h"
#include "MatrixSlice.h"
#include "MatrixSlicer.h"
#include "MatrixSlicerOnline.h"
#include "MatrixOnlineSchedulingStrategyFactory.h"
#include "common/ProblemStatement.h"
#include "common/MpiGuard.h"

#include <algorithm>
#include <iterator>
#include <stdexcept>

using namespace std;
using MatrixHelper::MatrixPair;

vector<MatrixSlice> MatrixOnlineScheduler::sliceDefinitions = std::vector<MatrixSlice>();
vector<MatrixSlice>::iterator MatrixOnlineScheduler::currentSliceDefinition = MatrixOnlineScheduler::sliceDefinitions.begin();
map<int, bool> MatrixOnlineScheduler::finishedSlaves = map<int, bool>();
vector<future<void>> MatrixOnlineScheduler::scheduleHandlers = vector<future<void>>();
mutex MatrixOnlineScheduler::schedulingMutex;

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

void MatrixOnlineScheduler::doDispatch()
{
    if (MpiGuard::getThreadSupport() == MPI_THREAD_MULTIPLE)
        MatrixScheduler::doDispatch();
    else
        throw runtime_error("\
MatrixOnlineScheduler needs MPI with MPI_THREAD_MULTIPLE support.\n\
Please rebuild MPI with enabled multiple thread support.");
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
        scheduleHandlers.push_back(async(launch::async,
            [&, requestingNode] () { schedule(requestingNode); }));
    }
}

void MatrixOnlineScheduler::schedule(const int node)
{
    const int lastWorkAmount = getLastWorkAmountFor(node);
    fetchResultsFrom(node, lastWorkAmount);
    sendNextSlicesTo(node);
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

int MatrixOnlineScheduler::getLastWorkAmountFor(const int node) const
{
    return schedulingStrategy->getLastWorkAmountFor(*this, node);
}

int MatrixOnlineScheduler::getNextWorkAmountFor(const int node) const
{
    return schedulingStrategy->getNextWorkAmountFor(*this, node);
}

vector<MatrixPair> MatrixOnlineScheduler::getNextWorkFor(
    const int node,
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

void MatrixOnlineScheduler::getWorkData(
    const int node,
    vector<MatrixPair>& work,
    int& workAmount)
{
    schedulingMutex.lock();
    vector<MatrixSlice>::iterator workStartingSliceDefinition = currentSliceDefinition;
    int remainingWorkAmount = getRemainingWorkAmount();
    workAmount = getNextWorkAmountFor(node);
    currentSliceDefinition += min(remainingWorkAmount, workAmount);
    schedulingMutex.unlock();
    work = getNextWorkFor(node, workStartingSliceDefinition, workAmount);
}

MatrixSlice& MatrixOnlineScheduler::getNextSliceDefinitionFor(const int node)
{
    for (auto& slice : sliceDefinitions)
        if (slice.getNodeId() == node)
            return slice;
    throw runtime_error("No next slice definition found.");
}

void MatrixOnlineScheduler::sendNextSlicesTo(const int node)
{
    int workAmount;
    vector<MatrixPair> work;
    getWorkData(node, work, workAmount);
    MatrixHelper::sendWorkAmountTo(communicator, node, workAmount);
    for (const auto& workPair : work)
    {
        if (workPair.first.empty() || workPair.second.empty())
            finishedSlaves[node] = true;
        MatrixHelper::sendMatrixTo(communicator, workPair.first, node);
        MatrixHelper::sendMatrixTo(communicator, workPair.second, node);
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

