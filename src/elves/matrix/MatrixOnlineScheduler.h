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
};
