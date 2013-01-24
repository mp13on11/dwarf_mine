#pragma once

#include "common/SchedulerTemplate.h"
#include "MonteCarloElf.h"
#include <functional>
#include <vector>
#include <mpi.h>

class MonteCarloElf;
class OthelloResult;

class MonteCarloScheduler: public SchedulerTemplate<MonteCarloElf>
{
public:
    MonteCarloScheduler(const std::function<ElfPointer()>& factory);
    virtual ~MonteCarloScheduler() = default;

    virtual void provideData(ProblemStatement& statement);
    virtual void outputData(ProblemStatement& statement);

protected:
    virtual bool hasData();
    virtual void doDispatch();

    OthelloState _state;
    OthelloResult _result;
    size_t _repetitions;
    MPI_Comm _scheduledCOMM;

private:
    void orchestrateCalculation();
    void calculateOnSlave();
    void calculate();
    void distribute();
    void collectResults();
	std::vector<OthelloResult> gatherResults();
};
