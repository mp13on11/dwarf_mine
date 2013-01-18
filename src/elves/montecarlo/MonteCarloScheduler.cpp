#include "MonteCarloScheduler.h"
#include <Elf.h>
#include <main/ProblemStatement.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <main/MpiHelper.h>
#include <mpi.h>

using namespace std;

struct MonteCarloScheduler::MonteCarloSchedulerImpl
{
    MonteCarloSchedulerImpl(MonteCarloScheduler* self) : self(self) {}

    void orchestrateCalculation();
    void calculateOnSlave();
    void calculateOnMaster();
    void distribute();
    void collectResults();

    void outputData(ProblemStatement& statement);
    bool hasData();
    void provideData(ProblemStatement& statement);

    // Reference to containing MonteCarloScheduler
    MonteCarloScheduler* self;
    OthelloState state;
    OthelloResult result;
};

MonteCarloScheduler::MonteCarloScheduler(const function<ElfPointer()>& factory) :
    SchedulerTemplate(factory), pImpl(new MonteCarloSchedulerImpl(this))
{
}

MonteCarloScheduler::~MonteCarloScheduler()
{
    delete pImpl;
}

void MonteCarloScheduler::provideData(ProblemStatement& statement)
{
    pImpl->provideData(statement);
}

bool MonteCarloScheduler::hasData()
{
    return pImpl->hasData();
}

void MonteCarloScheduler::outputData(ProblemStatement& statement)
{
    pImpl->outputData(statement);
}

void MonteCarloScheduler::doDispatch()
{
    if (MpiHelper::isMaster(rank))
    {
        pImpl->orchestrateCalculation();
    }
    else
    {
        pImpl->calculateOnSlave();
    }
}

void MonteCarloScheduler::MonteCarloSchedulerImpl::calculateOnSlave()
{
   result = self->elf().calculateBestMove(state);
}

inline void MonteCarloScheduler::MonteCarloSchedulerImpl::calculateOnMaster()
{
    calculateOnSlave();
}

void MonteCarloScheduler::MonteCarloSchedulerImpl::provideData(ProblemStatement& statement)
{
    statement.input->clear();
    statement.input->seekg(0);
    vector<Field> playfield;
    *(statement.input )>>playfield;
    state = OthelloState(playfield, Player::White);
}

void MonteCarloScheduler::MonteCarloSchedulerImpl::outputData(ProblemStatement& statement)
{
    *(statement.output) << result;
}

bool MonteCarloScheduler::MonteCarloSchedulerImpl::hasData()
{
    return true;
}

void MonteCarloScheduler::MonteCarloSchedulerImpl::orchestrateCalculation()
{
    distribute();
    calculateOnMaster();
    collectResults();
}


void MonteCarloScheduler::MonteCarloSchedulerImpl::distribute()
{
    size_t bufferSize = 0;
    if (MpiHelper::isMaster())
    {
        bufferSize = state.playfieldSideLength() * state.playfieldSideLength();
    }
    MPI::COMM_WORLD.Bcast(&bufferSize, 1, MPI::UNSIGNED, MpiHelper::MASTER);
    
    vector<Field> playfield(bufferSize);
    if (MpiHelper::isMaster())
    {
        playfield.assign(state.playfieldBuffer(), state.playfieldBuffer() + bufferSize);
    }
    
    auto MPI_FIELD = MPI::INT;
    MPI::COMM_WORLD.Bcast(playfield.data(), bufferSize, MPI_FIELD, MpiHelper::MASTER);
    state = OthelloState(playfield, Player::White);
}

void MonteCarloScheduler::MonteCarloSchedulerImpl::collectResults()
{
    size_t numberOfNodes = MpiHelper::numberOfNodes();
    vector<OthelloResult> results(numberOfNodes);
    // TODO OthelloResult - not defined for MPI

    MPI_Datatype MPI_OthelloResult;
    MPI_Datatype type[] = { MPI::INT, MPI::INT, MPI::UNSIGNED, MPI::UNSIGNED };
    int blocklen[] = { 1, 1, 1, 1 };
    MPI_Aint disp[] = { 
        0,
        sizeof(size_t),
        2 * sizeof(size_t),
        3 * sizeof(size_t)
    };

    MPI_Type_create_struct(4, blocklen, disp, type, &MPI_OthelloResult);
    MPI_Type_commit(&MPI_OthelloResult);
    MPI_Gather(
        &result, 1, MPI_OthelloResult, 
        results.data(), numberOfNodes, MPI_OthelloResult, MpiHelper::MASTER, MPI_COMM_WORLD);

    vector<OthelloResult> accumulatedResults;
    OthelloResult* bestMove;
    for (const auto& r : results)
    {
        OthelloResult* accumulated = nullptr;
        for (auto& a : accumulatedResults)
        {
            if (r.equalMove(a))
            {
                accumulated = &(a);
                break;
            }
        }
        if (accumulated == nullptr)
        {
            accumulatedResults.push_back(r);
            accumulated = &(accumulatedResults.back());
        }
        else
        {
            accumulated->visits += r.visits;
            accumulated->wins += r.wins;
        }
        if (bestMove == nullptr || bestMove->successRate() < accumulated->successRate())
        {
            bestMove = accumulated;
        }
    }
    result = *bestMove;
}

