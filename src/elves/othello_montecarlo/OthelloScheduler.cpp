#include "OthelloScheduler.h"
#include <Elf.h>
#include <common/ProblemStatement.h>
#include <Result.h>
#include <OthelloUtil.h>
#include <limits>
#include <utility>
#include <vector>
#include <iostream>
#include <memory>
#include <sstream>
#include <cmath>
#include <cassert>

using namespace std;

const Player DEFAULT_PLAYER = Player::White;
const size_t DEFAULT_SEED = 7337;
const size_t DEFAULT_REITERATIONS = 1000U;

OthelloScheduler::OthelloScheduler(const Communicator& communicator, const function<ElfPointer()>& factory) :
    SchedulerTemplate(communicator, factory),
    _repetitions(DEFAULT_REITERATIONS),
    _localRepetitions(0U),
    _commonSeed(DEFAULT_SEED)
{
}

void OthelloScheduler::provideData(std::istream& input)
{
    input.clear();
    input.seekg(0);
    input >>_commonSeed;
    input >>_repetitions;
    vector<Field> playfield;
    OthelloHelper::readPlayfieldFromStream(input, playfield);
    _state = State(playfield, DEFAULT_PLAYER);
}

void OthelloScheduler::generateData(const DataGenerationParameters& params)
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
    _state = State(playfield, DEFAULT_PLAYER);
}

bool OthelloScheduler::hasData() const
{
    return true;
}

void OthelloScheduler::outputData(std::ostream& output)
{
    cout<< "Move: ("<<_result.x<<", "<<_result.y<<")\n"
        << "Wins: "<<_result.wins<<"/"<<_result.visits<<" = " <<_result.successRate()<<endl;
    OthelloHelper::writeResultToStream(output, _result);
}

void OthelloScheduler::doSimpleDispatch()
{
    _localRepetitions = _repetitions;
    calculate();
}

void OthelloScheduler::doDispatch(BenchmarkResult nodeSet)
{
    if (communicator.isMaster())
    {
        distributeInput(nodeSet);
        calculate();
        collectResults(nodeSet);
    }
    else
    {
        collectInput(nodeSet);
        calculate();
        gatherResults(nodeSet);
    }
}

void OthelloScheduler::doDispatch()
{
   doDispatch(nodeSet);
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

void OthelloScheduler::calculate()
{
    if (_localRepetitions == 0)
        return;

    _result = elf().getBestMoveFor(_state, _localRepetitions, communicator.rank(), _commonSeed);
}

void registerResultToMPI(MPI_Datatype& type)
{
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

    MPI_Type_create_struct(4, elementLengths, elementDisplacements, elementTypes, &type);
    MPI_Type_commit(&type);
}

vector<Result> OthelloScheduler::gatherResults(BenchmarkResult nodeSet)
{
    size_t numberOfNodes = nodeSet.size();
    vector<Result> results(numberOfNodes);

    MPI_Datatype MPI_Result;
    registerResultToMPI(MPI_Result);
    
    if (communicator.isMaster())
    {
        int i = 0;
        for (const auto& node : nodeSet)
        {
            if (node.first == Communicator::MASTER_RANK)
            {
                results[0] = _result;
            }
            else
            {                
                communicator->Recv(results.data() + i , 1, MPI_Result, node.first, 0);
            }
            i++;
        }
    }
    else
    {
        communicator->Send(&_result, 1, MPI_Result, Communicator::MASTER_RANK, 0);
    }
    return results;
}

void OthelloScheduler::collectInput(BenchmarkResult /*nodeSet*/)
{
    unsigned long parameters[] = {0, 0, 0};
    communicator->Recv(parameters, 3, MPI::UNSIGNED_LONG, Communicator::MASTER_RANK, 0);

    size_t bufferSize = (size_t) parameters[0];
    _commonSeed =       (size_t) parameters[1];
    _localRepetitions = (size_t) parameters[2];

    Playfield playfield(bufferSize);

    auto MPI_FIELD = MPI::INT;
    communicator->Recv(playfield.data(), bufferSize, MPI_FIELD, Communicator::MASTER_RANK, 0);

    _state = State(playfield, Player::White);
}

void OthelloScheduler::distributeInput(BenchmarkResult nodeSet)
{
    size_t bufferSize = _state.playfieldSideLength() * _state.playfieldSideLength();
    vector<NodeRating> ratings;
    Rating ratingSum;
    tie(ratings, ratingSum) = weightRatings(nodeSet);

    auto MPI_FIELD = MPI::INT;
    const NodeRating* masterRating = nullptr;
    for (const auto& rating : ratings)
    {
        if (rating.first != Communicator::MASTER_RANK)
        {
            unsigned long parameters[] = { 
                bufferSize, 
                _commonSeed, 
                (size_t)round(_repetitions * rating.second / ratingSum) 
            };
            communicator->Send(parameters, 3, MPI::UNSIGNED_LONG, rating.first, 0);
            communicator->Send(_state.playfieldBuffer(), bufferSize, MPI_FIELD, rating.first, 0);
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



void OthelloScheduler::collectResults(BenchmarkResult nodeSet)
{
    vector<Result> results = gatherResults(nodeSet);

    vector<Result> accumulatedResults;
    Result* bestMove = nullptr;
    for (const auto& r : results)
    {
        Result* existingMove = nullptr;
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

void OthelloScheduler::doBenchmarkDispatch(int node )
{
    BenchmarkResult benchmarkNodeset;
    benchmarkNodeset[node] = 1;
    doDispatch(benchmarkNodeset);
}    
