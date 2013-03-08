#pragma once

#include "Matrix.h"
#include "MatrixScheduler.h"

#include <functional>
#include <vector>

class MatrixElf;
class MatrixSlice;

class MatrixOnlineScheduler: public MatrixScheduler
{
public:
    MatrixOnlineScheduler(const std::function<ElfPointer()>& factory);
    virtual ~MatrixOnlineScheduler();

    static const int slaves = 4;
    static bool finishedWorkers[4];

protected:
    std::vector<MatrixSlice> sliceDefinitions;
    std::vector<MatrixSlice>::const_iterator currentSliceDefinition;

    virtual void doDispatch();

    virtual void orchestrateCalculation();
    virtual void calculateOnSlave();

private:
    void sliceInput();
    void schedule();
    void fetchResultFrom(const NodeId node);
    void sendNextSlicesTo(const NodeId node);
    bool hasSlices() const;
    bool haveSlavesFinished() const;
    void calculateNextResult(
        Matrix<float>& result,
        Matrix<float>& left,
        Matrix<float>& right);
    void collectResults();
};
