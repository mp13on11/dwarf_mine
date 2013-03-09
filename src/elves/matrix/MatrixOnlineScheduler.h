#pragma once

#include "Matrix.h"
#include "MatrixScheduler.h"

#include <functional>
#include <vector>
#include <map>
#include <string>

class MatrixElf;
class MatrixSlice;

class MatrixOnlineScheduler: public MatrixScheduler
{
public:
    MatrixOnlineScheduler(const std::function<ElfPointer()>& factory);
    virtual ~MatrixOnlineScheduler();

    virtual void generateData(const DataGenerationParameters& params);

protected:
    virtual void doDispatch();

    virtual void orchestrateCalculation();
    virtual void calculateOnSlave();

private:
    static std::vector<MatrixSlice> sliceDefinitions;
    static std::vector<MatrixSlice>::iterator currentSliceDefinition;
    static std::map<NodeId, bool> finishedSlaves;
    std::string mode;

    void sliceInput();
    void schedule();
    void fetchResultFrom(const NodeId node);
    MatrixSlice& getNextSliceDefinitionFor(const NodeId node);
    void sendNextSlicesTo(const NodeId node);
    bool hasSlices() const;
    bool haveSlavesFinished() const;
    void calculateNextResult(
        Matrix<float>& result,
        Matrix<float>& left,
        Matrix<float>& right);
    void collectResults();
};
