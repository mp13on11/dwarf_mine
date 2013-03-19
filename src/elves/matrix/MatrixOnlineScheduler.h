#pragma once

#include "Matrix.h"
#include "MatrixScheduler.h"
#include "MatrixHelper.h"
#include "MatrixOnlineSchedulingStrategy.h"

#include <functional>
#include <vector>
#include <map>

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
    virtual void orchestrateCalculation();
    virtual void calculateOnSlave();

private:
    std::unique_ptr<MatrixOnlineSchedulingStrategy> schedulingStrategy;
    static std::vector<MatrixSlice> sliceDefinitions;
    static std::vector<MatrixSlice>::iterator currentSliceDefinition;
    static std::map<int, bool> finishedSlaves;
    std::vector<MatrixHelper::MatrixPair> workQueue;
    std::vector<Matrix<float>> resultQueue;

    void sliceInput();
    void schedule();
    void fetchResultsFrom(const int node, const int workAmount);
    int getWorkAmountFor(const int node) const;
    MatrixSlice& getNextSliceDefinitionFor(const int node);
    void sendNextSlicesTo(const int node, const int workAmount);
    void sendWorkAmountTo(const int node, const int workAmount);
    int getRemainingWorkAmount();
    bool hasSlices() const;
    bool haveSlavesFinished() const;

    bool hasToWork();
    void initiateCommunication() const;
    void sendResults();
    void receiveWork();
    void doWork();
};
