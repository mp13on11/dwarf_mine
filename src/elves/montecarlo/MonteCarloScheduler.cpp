#include "MonteCarloScheduler.h"
#include <Elf.h>
#include <common/ProblemStatement.h>
#include <common/MpiHelper.h>
#include <limits>
#include <utility>
#include <vector>
#include <iostream>
#include <memory>
#include <sstream>
#include <mpi.h>

using namespace std;

const Player DEFAULTPLAYER = Player::White;

MonteCarloScheduler::MonteCarloScheduler(const function<ElfPointer()>& factory) :
    SchedulerTemplate(factory), _repetitions(10U)
{
}

void MonteCarloScheduler::provideData(ProblemStatement& statement)
{
    statement.input->clear();
    statement.input->seekg(0);
    vector<Field> playfield;
    OthelloHelper::readPlayfieldFromStream(*(statement.input), playfield);
    _state = OthelloState(playfield, DEFAULTPLAYER);
}

bool MonteCarloScheduler::hasData()
{
    return true;
}

void MonteCarloScheduler::outputData(ProblemStatement& statement)
{
    OthelloHelper::writeResultToStream(*(statement.output), _result);
}

void MonteCarloScheduler::doDispatch()
{
    MPI_Group worldGroup;
    MPI_Group scheduledGroup;
    MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);
    int* ranks = new int[nodeSet.size()];
    int i = 0;
    for (const auto& node : nodeSet)
    {
        ranks[i] = node.first;
        i++;
    }
    MPI_Group_incl(worldGroup, nodeSet.size(), ranks, &scheduledGroup);
    MPI_Comm_create(MPI_COMM_WORLD, scheduledGroup, &_scheduledCOMM);

    if (MpiHelper::isMaster(rank))
    {
        orchestrateCalculation();
    }
    else
    {
        calculateOnSlave();
    }

    delete[] ranks;
    MPI_Comm_free(&_scheduledCOMM);
}

void MonteCarloScheduler::calculate()
{
    _result = elf().getBestMoveFor(_state, _repetitions);
}

vector<OthelloResult> MonteCarloScheduler::gatherResults()
{
    size_t numberOfNodes = nodeSet.size();
    vector<OthelloResult> results(numberOfNodes);

    MPI_Datatype MPI_OthelloResult;
    MPI_Datatype elementTypes[] = { 
        MPI::INT, 
        MPI::INT, 
        MPI::UNSIGNED, 
        MPI::UNSIGNED 
    };
    int elementLengths[] = { 
        1, 
        1, 
        1, 
        1 
    };
    MPI_Aint elementDisplacements[] = { 
        0,
        sizeof(size_t),
        2 * sizeof(size_t),
        3 * sizeof(size_t)
    };

    MPI_Type_create_struct(4, elementLengths, elementDisplacements, elementTypes, &MPI_OthelloResult);
    MPI_Type_commit(&MPI_OthelloResult);
    
    MPI_Gather(
        &_result, 1, MPI_OthelloResult, 
        results.data(), 1, MPI_OthelloResult, MpiHelper::MASTER, _scheduledCOMM);
    cout << "TEST" <<endl;
    return results;
}

void MonteCarloScheduler::calculateOnSlave()
{
    size_t paramters[2];
    MPI_Recv(paramters, 2, MPI::UNSIGNED, rank, 0, _scheduledCOMM, 0);

    size_t bufferSize = paramters[0];
    _repetitions = paramters[1];

    Playfield playfield(bufferSize);
    if (MpiHelper::isMaster())
    {
        playfield.assign(_state.playfieldBuffer(), _state.playfieldBuffer() + bufferSize);
    }
    
    auto MPI_FIELD = MPI::INT;
    MPI_Bcast(playfield.data(), bufferSize, MPI_FIELD, MpiHelper::MASTER, _scheduledCOMM);
    _state = OthelloState(playfield, Player::White);

    calculate();
    gatherResults();
}

void MonteCarloScheduler::orchestrateCalculation()
{
    cout << "Distribute "<< endl;
    distribute();
    cout << "Calculate "<< endl;
    calculate();
    cout << "Collect "<< endl;
    collectResults();
    cout << "Ready "<< endl;
}

pair<vector<NodeRating>, Rating> weightRatings(const BenchmarkResult& ratings)
{
    vector<NodeRating> positiveRatings;
    Rating ratingSum = 0;
    Rating ratingMax = 0;
    Rating ratingMin = numeric_limits<Rating>::max();
    for (const auto& rating : ratings)
    {
        ratingMax = max(ratingMax, rating.second);
        ratingMin = min(ratingMin, rating.second);
    }
    for (const auto& rating : ratings)
    {
        Rating positiveRating = ratingMin / rating.second;
        ratingSum += positiveRating;
        positiveRatings.emplace_back(rating.first, positiveRating);
    }
    return make_pair<vector<NodeRating>, Rating>(move(positiveRatings), move(ratingSum));
}
#include <iostream>
void MonteCarloScheduler::distribute()
{
    size_t bufferSize = _state.playfieldSideLength() * _state.playfieldSideLength();
    vector<NodeRating> ratings;
    Rating ratingSum;
    tie(ratings, ratingSum) = weightRatings(nodeSet);
    
    const NodeRating* masterRating = nullptr;
    for (const auto& rating : ratings)
    {
        if (rating.first != MpiHelper::MASTER)
        {
            size_t paramters[] = { bufferSize, (size_t)(_repetitions * rating.second / ratingSum) };
            MPI_Send(paramters, 2, MPI::UNSIGNED, rating.first, MpiHelper::MASTER, _scheduledCOMM);
        }
        else
        {
            masterRating = &rating;
        }
    }
    if (masterRating != nullptr)
    {
        _repetitions = (size_t)(_repetitions * masterRating->second / ratingSum);
    }

    //MPI::COMM_WORLD.Bcast(&bufferSize, 1, MPI::UNSIGNED, MpiHelper::MASTER);

    Playfield playfield(bufferSize);
    playfield.assign(_state.playfieldBuffer(), _state.playfieldBuffer() + bufferSize);
    
    auto MPI_FIELD = MPI::INT;
    MPI_Bcast(playfield.data(), bufferSize, MPI_FIELD, MpiHelper::MASTER, _scheduledCOMM);
    _state = OthelloState(playfield, Player::White);
}

void MonteCarloScheduler::collectResults()
{
    vector<OthelloResult> results = gatherResults();

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
    _result = *bestMove;
}

