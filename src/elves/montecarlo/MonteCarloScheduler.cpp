#include "MonteCarloScheduler.h"
#include <Elf.h>
#include <common/ProblemStatement.h>
#include <common/MpiHelper.h>
#include <OthelloResult.h>
#include <OthelloUtil.h>
#include <limits>
#include <utility>
#include <vector>
#include <iostream>
#include <memory>
#include <sstream>
#include <mpi.h>
#include <cmath>

using namespace std;

const Player DEFAULTPLAYER = Player::White;

MonteCarloScheduler::MonteCarloScheduler(const function<ElfPointer()>& factory) :
    SchedulerTemplate(factory), _repetitions(1000U), _localRepetitions(0U)
{
}

void MonteCarloScheduler::provideData(std::istream& input)
{
    input.clear();
    input.seekg(0);
    input >>_repetitions;
    vector<Field> playfield;
    OthelloHelper::readPlayfieldFromStream(input, playfield);
    _state = OthelloState(playfield, DEFAULTPLAYER);
}

void MonteCarloScheduler::generateData(const DataGenerationParameters& params)
{
    vector<Field> playfield = {
        F, F, F, F, F, F, F, F,
        F, F, F, F, F, F, F, F,
        F, F, F, F, F, F, F, F,
        F, F, F, W, B, F, F, F,      
        F, F, F, B, W, F, F, F,      
        F, F, F, F, F, F, F, F,
        F, F, F, F, F, F, F, F,
        F, F, F, F, F, F, F, F
    };
    _repetitions = params.monteCarloTrials;
    _state = OthelloState(playfield, DEFAULTPLAYER);
}

bool MonteCarloScheduler::hasData() const
{
    return true;
}

void MonteCarloScheduler::outputData(std::ostream& output)
{
    OthelloHelper::writeResultToStream(output, _result);
}

void MonteCarloScheduler::doSimpleDispatch()
{
    _localRepetitions = _repetitions;
    calculate();
}

void MonteCarloScheduler::doDispatch()
{
    if (MpiHelper::isMaster())
    {
        distributeInput();
        if (_localRepetitions != 0)
        {
            calculate();
        }
        collectResults();
    }
    else
    {
        collectInput();
        calculate();
        gatherResults();
    }
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

void MonteCarloScheduler::calculate()
{
    cout << "reiterations "<<_localRepetitions<<endl;
    _result = elf().getBestMoveFor(_state, _localRepetitions);
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
    
    if (MpiHelper::isMaster())
    {
        int i = 0;
        for (const auto& node : nodeSet)
        {
            if (MpiHelper::isMaster(node.first))
            {
                results[0] = _result;
            }
            else
            {
                
                MPI::COMM_WORLD.Recv(results.data() + i , 1, MPI_OthelloResult, node.first, 0);
            }
            i++;
        }
    }
    else
    {
        MPI::COMM_WORLD.Send(&_result, 1, MPI_OthelloResult, MpiHelper::MASTER, 0);
    }
    return results;
}

void MonteCarloScheduler::collectInput()
{
    unsigned long parameters[] = {0, 0};
    MPI::COMM_WORLD.Recv(parameters, 2, MPI::UNSIGNED_LONG, MpiHelper::MASTER, 0);

    size_t bufferSize = (size_t)parameters[0];
    _localRepetitions = (size_t)parameters[1];

    Playfield playfield(bufferSize);

    auto MPI_FIELD = MPI::INT;
    MPI::COMM_WORLD.Recv(playfield.data(), bufferSize, MPI_FIELD, MpiHelper::MASTER, 0);

    _state = OthelloState(playfield, Player::White);
}

void MonteCarloScheduler::distributeInput()
{
    size_t bufferSize = _state.playfieldSideLength() * _state.playfieldSideLength();
    vector<NodeRating> ratings;
    Rating ratingSum;
    tie(ratings, ratingSum) = weightRatings(nodeSet);

    auto MPI_FIELD = MPI::INT;
    const NodeRating* masterRating = nullptr;
    for (const auto& rating : ratings)
    {
        if (!MpiHelper::isMaster(rating.first))
        {
            unsigned long parameters[] = { bufferSize, (size_t)round(_repetitions * rating.second / ratingSum) };
            MPI::COMM_WORLD.Send(parameters, 2, MPI::UNSIGNED_LONG, rating.first, 0);
            MPI::COMM_WORLD.Send(_state.playfieldBuffer(), bufferSize, MPI_FIELD, rating.first, 0);
        }
        else
        {
            masterRating = &rating;
        }
    }
    if (masterRating != nullptr)
    {
        _localRepetitions = (size_t)(_repetitions * masterRating->second / ratingSum);
    }
    else
    {
        _localRepetitions = 0;
    }
}



void MonteCarloScheduler::collectResults()
{
    vector<OthelloResult> results = gatherResults();

    vector<OthelloResult> accumulatedResults;
    OthelloResult* bestMove = nullptr;
    for (const auto& r : results)
    {
        OthelloResult* existingMove = nullptr;
        for (auto& a : accumulatedResults)
        {
            if (r.equalMove(a))
            {
                existingMove = &(a);
                break;
            }
        }
        if (existingMove == nullptr)
        {
            accumulatedResults.push_back(r);
            existingMove = &(accumulatedResults.back());
        }
        else
        {
            existingMove->visits += r.visits;
            existingMove->wins += r.wins;
        }
        if (bestMove == nullptr || bestMove->successRate() < existingMove->successRate())
        {
            bestMove = existingMove;
        }
    }
    _result = *bestMove;
}

