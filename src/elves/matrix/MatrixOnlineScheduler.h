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
    virtual void collectResults(
        //const std::vector<MatrixSlice>& slices,
        Matrix<float>& result) const;

private:
    void sliceInput();
    void distributeToSlaves();
    bool hasSlices() const;
    bool haveSlavesFinished() const;
};
