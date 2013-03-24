#pragma once

#include "Matrix.h"
#include "MatrixScheduler.h"
#include "MatrixHelper.h"
#include "MatrixOnlineSchedulingStrategy.h"

#include <future>
#include <functional>
#include <vector>
#include <map>
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
    // Master
    std::unique_ptr<MatrixOnlineSchedulingStrategy> schedulingStrategy;
    static std::vector<MatrixSlice> sliceDefinitions;
    static std::vector<MatrixSlice>::iterator currentSliceDefinition;
    static std::map<int, bool> finishedSlaves;
    static std::mutex schedulingMutex;

    void sliceInput();
    void schedule();
    void scheduleWork();
    void sendNextWorkTo(const int node);
    void receiveResults();
    void receiveResultFrom(const int node);
    MatrixSlice& getNextSliceDefinitionFor(const int node);
    bool hasSlices() const;
    bool haveSlavesFinished() const;

    // Slave
    size_t maxWorkQueueSize;
    std::vector<MatrixHelper::MatrixPair> workQueue;
    std::condition_variable receiveWorkState;
    std::condition_variable doWorkState;
    std::mutex workMutex;
    bool receivedAllWork;
    
    void getWorkQueueSize();
    void receiveWork();
    Matrix<float> calculateNextResult();
    MatrixHelper::MatrixPair getNextWork();
    void sendResult(const Matrix<float>& result);
    void initiateWorkReceiving() const;
    void initiateResultSending() const;
    void initiateTransaction(const int tag) const;
    bool hasToReceiveWork();
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
