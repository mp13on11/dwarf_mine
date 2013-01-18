#pragma once

#include "main/SchedulerTemplate.h"

#include <functional>

class MatrixElf;

class MatrixScheduler: public SchedulerTemplate<MatrixElf>
{
public:
    MatrixScheduler(const std::function<ElfPointer()>& factory);
    virtual ~MatrixScheduler();

    virtual void provideData(ProblemStatement& statement);
    virtual void outputData(ProblemStatement& statement);

protected:
    virtual bool hasData();
    virtual void doDispatch();

private:
    struct MatrixSchedulerImpl;
    MatrixSchedulerImpl* pImpl;
};
