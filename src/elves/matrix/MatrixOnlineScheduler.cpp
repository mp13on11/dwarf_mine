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
#include <condition_variable>
#include <utility>

#include <iostream>
#include <fstream>

using namespace std;
using MatrixHelper::MatrixPair;

vector<MatrixSlice> MatrixOnlineScheduler::sliceDefinitions = std::vector<MatrixSlice>();
vector<MatrixSlice>::iterator MatrixOnlineScheduler::currentSliceDefinition = MatrixOnlineScheduler::sliceDefinitions.begin();
map<int, bool> MatrixOnlineScheduler::finishedSlaves = map<int, bool>();
mutex MatrixOnlineScheduler::schedulingMutex;

MatrixOnlineScheduler::MatrixOnlineScheduler(const Communicator& communicator, const function<ElfPointer()>& factory) :
    MatrixScheduler(communicator, factory)
{
    receivedAllWork = false;
    if (communicator.isMaster())
        for (size_t i = 0; i < communicator.size(); ++i)
            finishedSlaves[int(i)] = false;
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
Please use --mpi_thread_multiple and/or rebuild MPI with enabled multiple thread support.");
}

void MatrixOnlineScheduler::orchestrateCalculation()
{
    sliceInput();
    schedule();
}

void MatrixOnlineScheduler::sliceInput()
{
    result = Matrix<float>(left.rows(), right.columns());
    sliceDefinitions = schedulingStrategy->getSliceDefinitions(
        result, communicator.nodeSet());
    currentSliceDefinition = sliceDefinitions.begin();
}

ofstream file("/tmp/calc");
void MatrixOnlineScheduler::schedule()
{
    vector<future<void>> futures;
    maxWorkQueueSize = 3;
    futures.push_back(async(launch::async, [&]() { scheduleWork(); }));
    futures.push_back(async(launch::async, [&]() { receiveResults(); }));
    for (size_t i = 1; i < communicator.nodeSet().size(); ++i)
        MatrixHelper::sendWorkQueueSize(communicator, int(i), 3, int(Tags::workQueueSize));
    calculateOnSlave();
    waitFor(futures);
}

void MatrixOnlineScheduler::scheduleWork()
{
    vector<future<void>> futures;
    while (hasSlices() || !haveSlavesFinished())
    {
        const int requestingNode = MatrixHelper::waitForTransactionRequest(
            communicator, int(Tags::workRequest));
        futures.push_back(async(launch::async, [&, requestingNode]() {
            sendNextWorkTo(requestingNode); }));
    }
    waitFor(futures);
}

void MatrixOnlineScheduler::receiveResults()
{
    vector<future<void>> futures;
    while (!haveSlavesFinished())
    {
        const int requestingNode = MatrixHelper::waitForTransactionRequest(
            communicator, int(Tags::resultRequest));
        futures.push_back(async(launch::async, [&, requestingNode]() {
            receiveResultFrom(requestingNode); }));
    }
    waitFor(futures);
}

void MatrixOnlineScheduler::sendNextWorkTo(const int node)
{
    MatrixPair work;
    vector<MatrixSlice>::iterator workDefinition;
    schedulingMutex.lock();
    workDefinition = currentSliceDefinition;
    currentSliceDefinition++;
    schedulingMutex.unlock();
    if (workDefinition != sliceDefinitions.end())
    {
        workDefinition->setNodeId(node);
        work = sliceMatrices(*workDefinition);
    }
    else
    {
        work = MatrixPair(Matrix<float>(0, 0), Matrix<float>(0, 0));
        finishedSlaves[node] = true;
    }
    MatrixHelper::sendNextWork(communicator, work, node, int(Tags::exchangeWork));
}

void MatrixOnlineScheduler::receiveResultFrom(const int node)
{
    Matrix<float> nodeResult = MatrixHelper::receiveMatrixFrom(communicator, node, int(Tags::exchangeResult));
    if (result.empty()) return;
    MatrixSlice& sliceDefinition = getNextSliceDefinitionFor(node);
    sliceDefinition.injectSlice(nodeResult, result);
    sliceDefinition.setNodeId(-1);
}

MatrixSlice& MatrixOnlineScheduler::getNextSliceDefinitionFor(const int node)
{
    for (auto& slice : sliceDefinitions)
        if (slice.getNodeId() == node)
            return slice;
    throw runtime_error("No next slice definition found.");
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
    vector<future<void>> futures;
    getWorkQueueSize();
    futures.push_back(async(launch::async, [&]() { receiveWork(); }));
    while (hasToWork())
    {
        Matrix<float> result = calculateNextResult();
        futures.push_back(async(launch::async, [&]() { sendResult(move(result)); }));
    }
    waitFor(futures);
    file << "Salve done" << endl;
}

void MatrixOnlineScheduler::getWorkQueueSize()
{
    maxWorkQueueSize = MatrixHelper::receiveWorkQueueSize(communicator, Communicator::MASTER_RANK, int(Tags::workQueueSize));
}

void MatrixOnlineScheduler::receiveWork()
{
    file << "Receiver: Init workLock on workMutex." << endl;
    unique_lock<mutex> workLock(workMutex);
    while (hasToReceiveWork())
    {
        while (workQueue.size() >= maxWorkQueueSize)
        {
            file << "Receiver: Full workQueue; wait." << endl;
            receiveWorkState.wait(workLock);
        }
        file << "Receiver: rank = " <<  communicator.rank() << endl;
        file << "Receiver: Finished waiting." << endl;
        MatrixPair receivedWork = getNextWork();
        file << "Receiver: Try workMutex.lock()" << endl;
        workMutex.lock();
        file << "Receiver: Locked workMutex." << endl;
        workQueue.push_back(move(receivedWork));
        workMutex.unlock();
        file << "Receiver: Unlocked workMutex." << endl;
        doWorkState.notify_one();
        file << "Receiver: Notified Calc." << endl;
    }
    receivedAllWork = true;
}

Matrix<float> MatrixOnlineScheduler::calculateNextResult()
{
    file << "Calc: Init workLock on workMutex." << endl;
    unique_lock<mutex> workLock(workMutex);
    while (workQueue.empty())
    {
        file << "Calc: Empty workQueue; wait." << endl;
        doWorkState.wait(workLock);
    }
    file << "Calc: Finished waiting." << endl;
    file << "Calc: Try workMutex.lock()" << endl;
    workMutex.lock();
    file << "Calc: Locked workMutex." << endl;
    MatrixPair work = move(workQueue.back());
    workQueue.pop_back();
    workMutex.unlock();
    file << "Calc: Unlocked workMutex." << endl;
    receiveWorkState.notify_one();
    file << "Calc: Notified receiver." << endl;
    return elf().multiply(work.first, work.second);
}

MatrixPair MatrixOnlineScheduler::getNextWork()
{
    initiateWorkReceiving();
    return MatrixHelper::getNextWork(communicator, Communicator::MASTER_RANK, int(Tags::exchangeWork));
}

void MatrixOnlineScheduler::sendResult(const Matrix<float>& result)
{
    initiateResultSending();
    MatrixHelper::sendMatrixTo(
        communicator,
        result,
        Communicator::MASTER_RANK,
        int(Tags::exchangeResult));
}

void MatrixOnlineScheduler::initiateWorkReceiving() const
{
    initiateTransaction(int(Tags::workRequest));
}

void MatrixOnlineScheduler::initiateResultSending() const
{
    initiateTransaction(int(Tags::resultRequest));
}

void MatrixOnlineScheduler::initiateTransaction(const int tag) const
{
    MatrixHelper::requestTransaction(
        communicator,
        communicator.rank(),
        Communicator::MASTER_RANK,
        tag);
}

bool MatrixOnlineScheduler::hasToReceiveWork()
{
    return workQueue.empty()
        || !workQueue[workQueue.size() - 1].first.empty()
        || !workQueue[workQueue.size() - 1].second.empty();
}

bool MatrixOnlineScheduler::hasToWork()
{
    return !workQueue.empty() || !receivedAllWork;
}

void MatrixOnlineScheduler::waitFor(vector<future<void>>& futures)
{
    #pragma omp parallel for
    for (size_t i = 0; i < futures.size(); ++i)
        futures[i].wait();
}

