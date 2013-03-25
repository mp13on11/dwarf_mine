#pragma once

#include "Matrix.h"
#include "MatrixScheduler.h"
#include "MatrixHelper.h"
#include "MatrixOnlineSchedulingStrategy.h"

#include <future>
#include <functional>
#include <vector>
#include <deque>
#include <mutex>

class MatrixElf;
class MatrixSlice;

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
    static std::mutex fileMutex;

    // Master
    std::unique_ptr<MatrixOnlineSchedulingStrategy> schedulingStrategy;
    static std::vector<MatrixSlice> sliceDefinitions;
    static std::vector<MatrixSlice>::iterator currentSliceDefinition;
    static std::mutex schedulingMutex;

    void sliceInput();
    void schedule();
    void scheduleWork();
    void sendNextWorkTo(const int node);
    void receiveResults();
    void receiveResultFrom(const int node);
    MatrixSlice& getNextSliceDefinitionFor(const int node);
    size_t numberOfTransactions() const;

    // Slave
    size_t maxWorkQueueSize;
    std::vector<MatrixHelper::MatrixPair> workQueue;
    std::deque<Matrix<float>> resultQueue;
    std::condition_variable receiveWorkState;
    std::condition_variable sendResultsState;
    std::condition_variable doWorkState;
    std::mutex workMutex;
    std::mutex resultMutex;
    bool receivedAllWork;
    bool processedAllWork;
    
    void getWorkQueueSize();
    void receiveWork();
    void sendResults();
    Matrix<float> calculateNextResult();
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
