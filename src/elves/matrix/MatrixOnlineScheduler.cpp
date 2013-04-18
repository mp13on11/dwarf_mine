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

using namespace std;
using MatrixHelper::MatrixPair;

vector<MatrixSlice> MatrixOnlineScheduler::sliceDefinitions = std::vector<MatrixSlice>();
vector<SliceContainer> MatrixOnlineScheduler::sliceContainers = std::vector<SliceContainer>();
vector<SliceContainer>::iterator MatrixOnlineScheduler::currentSliceContainer = MatrixOnlineScheduler::sliceContainers.begin();
mutex MatrixOnlineScheduler::schedulingMutex;
int MatrixOnlineScheduler::sliceNodeIdNone = -1;

MatrixOnlineScheduler::MatrixOnlineScheduler(const Communicator& communicator, const function<ElfPointer()>& factory) :
    MatrixScheduler(communicator, factory)
{
    receivedAllWork = false;
    processedAllWork = false;
}

MatrixOnlineScheduler::~MatrixOnlineScheduler()
{
}

void MatrixOnlineScheduler::configureWith(const Configuration& config)
{
    schedulingStrategy = MatrixOnlineSchedulingStrategyFactory::getStrategy(config.schedulingStrategy());
    maxWorkQueueSize = schedulingStrategy->getWorkQueueSize();
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
    generateSlicePairs();
    schedule();
}

void MatrixOnlineScheduler::sliceInput()
{
    result = Matrix<float>(left.rows(), right.columns());
    sliceDefinitions = schedulingStrategy->getSliceDefinitions(
        result, communicator.nodeSet());
}

void MatrixOnlineScheduler::generateSlicePairs()
{
    SliceContainer container;
    for (auto& sliceDefinition : sliceDefinitions)
    {
        container.slicePair = sliceMatrices(sliceDefinition);
        container.sliceDefinition = sliceDefinition;
        sliceContainers.push_back(container);
    }
    currentSliceContainer = sliceContainers.begin();
} 

void MatrixOnlineScheduler::schedule()
{
    vector<future<void>> futures;
    futures.push_back(async(launch::async, [&]() { scheduleWork(); }));
    futures.push_back(async(launch::async, [&]() { receiveResults(); }));
    for (size_t i = 1; i < communicator.nodeSet().size(); ++i)
        futures.push_back(async(launch::async, [&, i] () {
            MatrixHelper::sendWorkQueueSize(communicator, int(i), maxWorkQueueSize, int(Tags::workQueueSize)); }));
    calculateOnSlave();
    waitFor(futures);
}

void MatrixOnlineScheduler::scheduleWork()
{
    vector<future<void>> futures;
    size_t sentWork = 0;
    while (sentWork != amountOfWork())
    {
        const int requestingNode = MatrixHelper::waitForTransactionRequest(
            communicator, int(Tags::workRequest));
        futures.push_back(async(launch::async, [&, requestingNode]() {
            sendNextWorkTo(requestingNode); }));
        sentWork++;
    }
    waitFor(futures);
}

void MatrixOnlineScheduler::receiveResults()
{
    vector<future<void>> futures;
    size_t receivedResults = 0;
    while (receivedResults != amountOfResults())
    {
        const int requestingNode = MatrixHelper::waitForTransactionRequest(
            communicator, int(Tags::resultRequest));
        futures.push_back(async(launch::async, [&, requestingNode]() {
            receiveResultFrom(requestingNode); }));
        receivedResults++;
    }
    waitFor(futures);
}

void MatrixOnlineScheduler::sendNextWorkTo(const int node)
{
    vector<SliceContainer>::iterator workDefinition = getNextWorkDefinition();
    if (workDefinition != sliceContainers.end())
    {
        (*workDefinition).sliceDefinition.setNodeId(node);
        MatrixHelper::isendNextWork(communicator, workDefinition->slicePair, node, int(Tags::exchangeWork));
    }
    else
        MatrixHelper::isendNextWork(communicator, MatrixPair(Matrix<float>(), Matrix<float>()), node, int(Tags::exchangeWork));
}

vector<SliceContainer>::iterator MatrixOnlineScheduler::getNextWorkDefinition()
{
    vector<SliceContainer>::iterator workDefinition;
    schedulingMutex.lock();
    workDefinition = currentSliceContainer;
    if (currentSliceContainer != sliceContainers.end())
        currentSliceContainer++;
    schedulingMutex.unlock();
    return move(workDefinition);
}

void MatrixOnlineScheduler::receiveResultFrom(const int node)
{
    Matrix<float> nodeResult = MatrixHelper::receiveMatrixFrom(communicator, node, int(Tags::exchangeResult));
    MatrixSlice& slice = getNextSliceDefinitionFor(node);
    slice.injectSlice(nodeResult, result);
    slice.setNodeId(sliceNodeIdNone);
}

MatrixSlice& MatrixOnlineScheduler::getNextSliceDefinitionFor(const int node)
{
    for (auto& slice : sliceContainers)
        if (slice.sliceDefinition.getNodeId() == node)
            return slice.sliceDefinition;
    throw runtime_error("No next slice definition found.");
}

size_t MatrixOnlineScheduler::amountOfWork() const
{
    return sliceDefinitions.size() + communicator.nodeSet().size();
}

size_t MatrixOnlineScheduler::amountOfResults() const
{
    return sliceDefinitions.size();
}

void MatrixOnlineScheduler::calculateOnSlave()
{
    vector<future<void>> futures;
    if (!communicator.isMaster())
        getWorkQueueSize();
    futures.push_back(async(launch::async, [&]() { receiveWork(); }));
    futures.push_back(async(launch::async, [&]() { sendResults(); }));
    while (hasToWork())
        doWork();
    waitFor(futures);
}

void MatrixOnlineScheduler::getWorkQueueSize()
{
    maxWorkQueueSize = MatrixHelper::receiveWorkQueueSize(communicator, Communicator::MASTER_RANK, int(Tags::workQueueSize));
}

void MatrixOnlineScheduler::doWork()
{
    Matrix<float> result = calculateNextResult();
    if (result.empty())
        processedAllWork = true;
    else
    {
        resultMutex.lock();
        resultQueue.push_back(result);
        resultMutex.unlock();
    }
    sendResultsState.notify_one();
}

Matrix<float> MatrixOnlineScheduler::calculateNextResult()
{
    unique_lock<mutex> workLock(workMutex);
    while (workQueue.empty())
        doWorkState.wait(workLock);
    MatrixPair work = move(workQueue.front());
    workQueue.pop_front();
    workMutex.unlock();
    receiveWorkState.notify_one();
    return move(elf().multiply(work.first, work.second));
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
        if (receivedWork.first.empty() || receivedWork.second.empty())
            receivedAllWork = true;
        workMutex.unlock();
        doWorkState.notify_one();
    }
}

void MatrixOnlineScheduler::sendResults()
{
    unique_lock<mutex> resultLock(resultMutex);
    while (hasToSendResults())
    {
        while (resultQueue.empty())
        {
            sendResultsState.wait(resultLock);
            if (resultQueue.empty() && processedAllWork)
            {
                resultMutex.unlock();
                return;
            }
        }
        Matrix<float> result = move(resultQueue.front());
        resultQueue.pop_front();
        resultMutex.unlock();
        sendResult(result);
    }
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
    return !receivedAllWork;
}

bool MatrixOnlineScheduler::hasToSendResults()
{
    return !resultQueue.empty() || !processedAllWork;
}

bool MatrixOnlineScheduler::hasToWork()
{
    return !workQueue.empty() || !receivedAllWork;
}

void MatrixOnlineScheduler::waitFor(vector<future<void>>& futures)
{
    #pragma omp parallel for
    for (size_t i = 0; i < futures.size(); ++i)
    {
        futures[i].wait();
        if (!futures[i].valid())
            throw future_error(future_errc::no_state);
    }
}

