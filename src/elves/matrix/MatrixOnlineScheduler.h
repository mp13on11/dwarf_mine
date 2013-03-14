#pragma once

#include "Matrix.h"
#include "MatrixScheduler.h"
#include "MatrixHelper.h"
#include "MatrixOnlineSchedulingStrategy.h"

#include <functional>
#include <vector>
#include <map>
#include <mutex>

class MatrixElf;
class MatrixSlice;

class MatrixOnlineScheduler: public MatrixScheduler
{
public:
    MatrixOnlineScheduler(const std::function<ElfPointer()>& factory);
    virtual ~MatrixOnlineScheduler();

    virtual void configureWith(const Configuration& config);
    virtual void generateData(const DataGenerationParameters& params);

    int getRemainingWorkAmount() const;

protected:
    virtual void orchestrateCalculation();
    virtual void calculateOnSlave();

private:
    std::unique_ptr<MatrixOnlineSchedulingStrategy> schedulingStrategy;
    static std::vector<MatrixSlice> sliceDefinitions;
    static std::vector<MatrixSlice>::iterator currentSliceDefinition;
    static std::map<NodeId, bool> finishedSlaves;
    static std::mutex schedulingMutex;
    std::vector<MatrixHelper::MatrixPair> workQueue;
    std::vector<Matrix<float>> resultQueue;

    void sliceInput();
    void schedule();
    void schedule(const NodeId node);
    void fetchResultsFrom(const NodeId node, const int workAmount);
    int getLastWorkAmountFor(const NodeId node) const;
    int getNextWorkAmountFor(const NodeId node) const;
    std::vector<MatrixHelper::MatrixPair> getNextWorkFor(
        const NodeId node,
        const int workAmount);
    MatrixSlice& getNextSliceDefinitionFor(const NodeId node);
    void sendNextSlicesTo(const NodeId node);
    bool hasSlices() const;
    bool haveSlavesFinished() const;

    bool hasToWork();
    void initiateCommunication() const;
    void sendResults();
    void receiveWork();
    void doWork();
};
