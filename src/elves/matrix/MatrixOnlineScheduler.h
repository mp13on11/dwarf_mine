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

    static int slices;
    static const int slaves = 3;
    static bool finishedWorkers[4];

protected:
    Matrix<float> left;
    Matrix<float> right;
    Matrix<float> result;

    virtual void doDispatch();

    virtual void orchestrateCalculation();
    virtual void calculateOnSlave();
    virtual void calculateOnMaster();
    virtual void collectResults(
        //const std::vector<MatrixSlice>& slices,
        Matrix<float>& result) const;

private:
    void sliceInput();
    void distributeToSlaves();
    bool hasSlices() const;
    bool haveSlavesFinished() const;
};
