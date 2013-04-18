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

#pragma once

#include "Matrix.h"
#include "MatrixScheduler.h"
#include "MatrixHelper.h"
#include "MatrixOnlineSchedulingStrategy.h"
#include "MatrixSlice.h"

#include <future>
#include <functional>
#include <vector>
#include <deque>
#include <mutex>

class MatrixElf;
class MatrixSlice;

struct SliceContainer
{
    MatrixHelper::MatrixPair slicePair;
    MatrixSlice sliceDefinition;
};

class MatrixOnlineScheduler: public MatrixScheduler
{
public:
    MatrixOnlineScheduler(const Communicator& communicator, const std::function<ElfPointer()>& factory);
    virtual ~MatrixOnlineScheduler();

    virtual void configureWith(const Configuration& config);
    virtual void generateData(const DataGenerationParameters& params);

protected:
    virtual void doDispatch();
    virtual void orchestrateCalculation();
    virtual void calculateOnSlave();

private:
    // Master
    std::unique_ptr<MatrixOnlineSchedulingStrategy> schedulingStrategy;
    static std::vector<MatrixSlice> sliceDefinitions;
    static std::vector<SliceContainer> sliceContainers;
    static std::vector<SliceContainer>::iterator currentSliceContainer;
    static std::mutex schedulingMutex;
    static int sliceNodeIdNone;

    void sliceInput();
    void generateSlicePairs();
    void schedule();
    void scheduleWork();
    void receiveResults();
    void sendNextWorkTo(const int node);
    std::vector<SliceContainer>::iterator getNextWorkDefinition();
    void receiveResultFrom(const int node);
    MatrixSlice& getNextSliceDefinitionFor(const int node);
    size_t amountOfWork() const;
    size_t amountOfResults() const;

    // Slave
    size_t maxWorkQueueSize;
    std::deque<MatrixHelper::MatrixPair> workQueue;
    std::deque<Matrix<float>> resultQueue;
    std::condition_variable receiveWorkState;
    std::condition_variable sendResultsState;
    std::condition_variable doWorkState;
    std::mutex workMutex;
    std::mutex resultMutex;
    bool receivedAllWork;
    bool processedAllWork;
    
    void getWorkQueueSize();
    void doWork();
    Matrix<float> calculateNextResult();
    void receiveWork();
    void sendResults();
    MatrixHelper::MatrixPair getNextWork();
    void sendResult(const Matrix<float>& result);
    void initiateWorkReceiving() const;
    void initiateResultSending() const;
    void initiateTransaction(const int tag) const;
    bool hasToReceiveWork();
    bool hasToSendResults();
    bool hasToWork();

    // Utilities
    void waitFor(std::vector<std::future<void>>& futures);
    
    enum class Tags : int
    {
        workQueueSize = 1,
        workRequest,
        resultRequest,
        exchangeWork,
        exchangeResult
    };
