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

    int getRemainingWorkAmount() const;

protected:
    virtual void doDispatch();
    virtual void orchestrateCalculation();
    virtual void calculateOnSlave();

private:
    std::unique_ptr<MatrixOnlineSchedulingStrategy> schedulingStrategy;
    static std::vector<MatrixSlice> sliceDefinitions;
    static std::vector<MatrixSlice>::iterator currentSliceDefinition;
    static std::map<int, bool> finishedSlaves;
    static std::vector<std::future<void>> scheduleHandlers;
    static std::mutex schedulingMutex;
    std::vector<MatrixHelper::MatrixPair> workQueue;
    std::vector<Matrix<float>> resultQueue;

    void sliceInput();
    void schedule();
    void schedule(const int node);
    void fetchResultsFrom(const int node, const int workAmount);
    int getLastWorkAmountFor(const int node) const;
    int getNextWorkAmountFor(const int node) const;
    std::vector<MatrixHelper::MatrixPair> getNextWorkFor(
        const int node,
        std::vector<MatrixSlice>::iterator& workSlice,
        const int workAmount);
    void getWorkData(
        const int node,
        std::vector<MatrixHelper::MatrixPair>& work,
        int& workAmount);
    MatrixSlice& getNextSliceDefinitionFor(const int node);
    void sendNextSlicesTo(const int node);
    bool hasSlices() const;
    bool haveSlavesFinished() const;

    bool hasToWork();
    void initiateCommunication() const;
    void sendResults();
    void receiveWork();
    void doWork();
};
