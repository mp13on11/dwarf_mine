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

mutex MatrixOnlineScheduler::fileMutex;

vector<MatrixSlice> MatrixOnlineScheduler::sliceDefinitions = std::vector<MatrixSlice>();
vector<MatrixSlice>::iterator MatrixOnlineScheduler::currentSliceDefinition = MatrixOnlineScheduler::sliceDefinitions.begin();
map<int, bool> MatrixOnlineScheduler::finishedSlaves = map<int, bool>();
mutex MatrixOnlineScheduler::schedulingMutex;

MatrixOnlineScheduler::MatrixOnlineScheduler(const Communicator& communicator, const function<ElfPointer()>& factory) :
    MatrixScheduler(communicator, factory)
{
    receivedAllWork = false;
    if (communicator.isMaster())
        for (size_t i = 0; i < communicator.nodeSet().size(); ++i)
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

ofstream fileBLA("/tmp/calc2");
void MatrixOnlineScheduler::schedule()
{
    vector<future<void>> futures;
    maxWorkQueueSize = 1;
    futures.push_back(async(launch::async, [&]() { scheduleWork(); }));
    futures.push_back(async(launch::async, [&]() { receiveResults(); }));
fileMutex.lock();
    fileBLA << "Sending work sizes" << endl;
fileMutex.unlock();
    for (size_t i = 1; i < communicator.nodeSet().size(); ++i)
        MatrixHelper::sendWorkQueueSize(communicator, int(i), 1, int(Tags::workQueueSize));
fileMutex.lock();
    fileBLA << "Start calculating" << endl;
fileMutex.unlock();
    calculateOnSlave();
fileMutex.lock();
    fileBLA << "Done calculating" << endl;
fileMutex.unlock();
    waitFor(futures);
}

void MatrixOnlineScheduler::scheduleWork()
{
    vector<future<void>> futures;
    while (hasSlices() || !haveSlavesFinished())
    {
fileMutex.lock();
        fileBLA << "Waiting for work requests." << endl;
fileMutex.unlock();
        const int requestingNode = MatrixHelper::waitForTransactionRequest(
            communicator, int(Tags::workRequest));
        futures.push_back(async(launch::async, [&, requestingNode]() {
            sendNextWorkTo(requestingNode); }));
    }
fileMutex.lock();
    fileBLA << "Done waiting for work requests." << endl;
fileMutex.unlock();
    waitFor(futures);
fileMutex.lock();
    fileBLA << "Shutting down work sender." << endl;
fileMutex.unlock();
}

void MatrixOnlineScheduler::receiveResults()
{
    vector<future<void>> futures;
    while (!haveSlavesFinished())
    {
fileMutex.lock();
        fileBLA << "Waiting for results." << endl;
fileMutex.unlock();
        const int requestingNode = MatrixHelper::waitForTransactionRequest(
            communicator, int(Tags::resultRequest));
        futures.push_back(async(launch::async, [&, requestingNode]() {
            receiveResultFrom(requestingNode); }));
    }
fileMutex.lock();
    fileBLA << "Done waiting for results." << endl;
fileMutex.unlock();
    waitFor(futures);
fileMutex.lock();
    fileBLA << "Shutting down result receiver." << endl;
fileMutex.unlock();
}

void MatrixOnlineScheduler::sendNextWorkTo(const int node)
{
    MatrixPair work;
    vector<MatrixSlice>::iterator workDefinition;
    schedulingMutex.lock();
    workDefinition = currentSliceDefinition;
    if (currentSliceDefinition != sliceDefinitions.end())
        currentSliceDefinition++;
    schedulingMutex.unlock();
    if (workDefinition != sliceDefinitions.end())
    {
        workDefinition->setNodeId(node);
        work = sliceMatrices(*workDefinition);
    }
    else
        work = MatrixPair(Matrix<float>(0, 0), Matrix<float>(0, 0));
    MatrixHelper::sendNextWork(communicator, work, node, int(Tags::exchangeWork));
}

void MatrixOnlineScheduler::receiveResultFrom(const int node)
{
    Matrix<float> nodeResult = MatrixHelper::receiveMatrixFrom(communicator, node, int(Tags::exchangeResult));
    if (nodeResult.empty())
    {
fileMutex.lock();
        fileBLA << "Slave " << node << " finished." << endl;
fileMutex.unlock();
        finishedSlaves[node] = true;
        return;
    }
    MatrixSlice& slice = getNextSliceDefinitionFor(node);
fileMutex.lock();
    fileBLA << "Inject using slice (" << slice.getRows() << "x" << slice.getColumns() << ") @ (" << slice.getStartY() << "x" << slice.getStartX() << ") with result matrix (" << nodeResult.rows() << "x" << nodeResult.columns() << ")." << endl;
fileMutex.unlock();
    slice.injectSlice(nodeResult, result);
    slice.setNodeId(-1);
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

ofstream file("/tmp/calc");
void MatrixOnlineScheduler::calculateOnSlave()
{
    vector<future<void>> futures;
    if (communicator.rank() != Communicator::MASTER_RANK)
        getWorkQueueSize();
    futures.push_back(async(launch::async, [&]() { receiveWork(); }));
    futures.push_back(async(launch::async, [&]() { sendResults(); }));
    while (hasToWork())
    {
fileMutex.lock();
        file << communicator.rank() << "Working." << endl;
fileMutex.unlock();
        Matrix<float> result = calculateNextResult();
fileMutex.lock();
        file << communicator.rank() << "Result: (" << result.rows() << "x" << result.columns() << ")" << endl;
fileMutex.unlock();
        resultMutex.lock();
        resultQueue.push_back(result);
        resultMutex.unlock();
        sendResultsState.notify_one();
    }
fileMutex.lock();
    file << communicator.rank() << "Slave done" << endl;
fileMutex.unlock();
    waitFor(futures);
}

void MatrixOnlineScheduler::getWorkQueueSize()
{
    maxWorkQueueSize = MatrixHelper::receiveWorkQueueSize(communicator, Communicator::MASTER_RANK, int(Tags::workQueueSize));
}

void MatrixOnlineScheduler::receiveWork()
{
    unique_lock<mutex> workLock(workMutex);
    while (hasToReceiveWork())
    {
        while (workQueue.size() >= maxWorkQueueSize)
            receiveWorkState.wait(workLock);
        MatrixPair receivedWork = getNextWork();
        workQueue.push_back(move(receivedWork));
        workMutex.unlock();
        doWorkState.notify_one();
    }
    receivedAllWork = true;
}

void MatrixOnlineScheduler::sendResults()
{
    unique_lock<mutex> resultLock(resultMutex);
    while (hasToSendResults())
    {
        while (resultQueue.empty())
            sendResultsState.wait(resultLock);
        Matrix<float> result = move(resultQueue.front());
        resultQueue.pop_front();
        resultMutex.unlock();
        sendResult(result);
    }
}

Matrix<float> MatrixOnlineScheduler::calculateNextResult()
{
    unique_lock<mutex> workLock(workMutex);
    while (workQueue.empty())
        doWorkState.wait(workLock);
    MatrixPair work = move(workQueue.back());
    workQueue.pop_back();
    workMutex.unlock();
    receiveWorkState.notify_one();
fileMutex.lock();
    file << communicator.rank() << "Working on left matrix: (" << work.first.rows() << "x" << work.first.columns() << ")" << endl;
    file << communicator.rank() << "Working on right matrix: (" << work.second.rows() << "x" << work.second.columns() << ")" << endl;
fileMutex.unlock();
    return move(elf().multiply(work.first, work.second));
}

MatrixPair MatrixOnlineScheduler::getNextWork()
{
    initiateWorkReceiving();
    return MatrixHelper::getNextWork(communicator, Communicator::MASTER_RANK, int(Tags::exchangeWork));
}

void MatrixOnlineScheduler::sendResult(const Matrix<float>& result)
{
fileMutex.lock();
    file << communicator.rank() << "Sending result (" << result.rows() << "x" << result.columns() << ")" << endl;
fileMutex.unlock();
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
        || !workQueue.back().first.empty()
        || !workQueue.back().second.empty();
}

bool MatrixOnlineScheduler::hasToSendResults()
{
    return !resultQueue.empty() || hasToWork();
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

