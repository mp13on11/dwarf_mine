#include "MonteCarloScheduler.h"
#include <Elf.h>
#include <common/ProblemStatement.h>
#include <OthelloResult.h>
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

MonteCarloScheduler::MonteCarloScheduler(const Communicator& communicator, const function<ElfPointer()>& factory) :
    SchedulerTemplate(communicator, factory),
    _repetitions(DEFAULT_REITERATIONS),
    _localRepetitions(0U),
    _commonSeed(DEFAULT_SEED)
{
}

void MonteCarloScheduler::provideData(std::istream& input)
{
    input.clear();
    input.seekg(0);
    input >>_commonSeed;
    input >>_repetitions;
    vector<Field> playfield;
    OthelloHelper::readPlayfieldFromStream(input, playfield);
    _state = OthelloState(playfield, DEFAULT_PLAYER);
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
    _state = OthelloState(playfield, DEFAULT_PLAYER);
}

bool MonteCarloScheduler::hasData() const
{
    return true;
}

void MonteCarloScheduler::outputData(std::ostream& output)
{
    cout<< "Move: ("<<_result.x<<", "<<_result.y<<")\n"
        << "Wins: "<<_result.wins<<"/"<<_result.visits<<" = " <<_result.successRate()<<endl;
    OthelloHelper::writeResultToStream(output, _result);
}

void MonteCarloScheduler::doSimpleDispatch()
{
    _localRepetitions = _repetitions;
    calculate();
}

void MonteCarloScheduler::doDispatch(BenchmarkResult nodeSet)
{
    distributeInput();
    calculate();

    if (communicator.isMaster())
    {
        collectResults(nodeSet);
    }
    else
    {
        gatherResults(nodeSet);
    }
}

void MonteCarloScheduler::doDispatch()
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

void MonteCarloScheduler::calculate()
{
    if (_localRepetitions == 0)
        return;

    _result = elf().getBestMoveFor(_state, _localRepetitions, communicator.rank(), _commonSeed);
}

void registerOthelloResultToMPI(MPI_Datatype& type)
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

vector<OthelloResult> MonteCarloScheduler::gatherResults(BenchmarkResult nodeSet)
{
    size_t numberOfNodes = nodeSet.size();
    vector<OthelloResult> results(numberOfNodes);

    MPI_Datatype MPI_OthelloResult;
    registerOthelloResultToMPI(MPI_OthelloResult);
    
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
                communicator->Recv(results.data() + i , 1, MPI_OthelloResult, node.first, 0);
            }
            i++;
        }
    }
    else
    {
        communicator->Send(&_result, 1, MPI_OthelloResult, Communicator::MASTER_RANK, 0);
    }
    return results;
}

void MonteCarloScheduler::distributeInput()
{
    size_t bufferSize = distributeCommonParameters();
    distributePlayfield(bufferSize);
}

size_t MonteCarloScheduler::distributeCommonParameters()
{
    size_t bufferSize = _state.playfieldSideLength() * _state.playfieldSideLength();
    unsigned long commonParameters[] = {bufferSize, _commonSeed, _repetitions};

    communicator->Bcast(commonParameters, 3, MPI::UNSIGNED_LONG, Communicator::MASTER_RANK);

    bufferSize = commonParameters[0];
    _commonSeed = commonParameters[1];
    _repetitions = commonParameters[2];
    _localRepetitions = _repetitions * communicator.weight();

    return bufferSize;
}

void MonteCarloScheduler::distributePlayfield(size_t bufferSize)
{
    Playfield playfield;

    if (communicator.isMaster())
    {
        playfield = Playfield(
                _state.playfieldBuffer(), _state.playfieldBuffer() + bufferSize
            );
    }
    else
    {
        playfield = Playfield(bufferSize);
    }

    auto MPI_FIELD = MPI::INT;
    communicator->Bcast(playfield.data(), bufferSize, MPI_FIELD, Communicator::MASTER_RANK);

    _state = OthelloState(playfield, Player::White);
}



void MonteCarloScheduler::collectResults(BenchmarkResult nodeSet)
{
    vector<OthelloResult> results = gatherResults(nodeSet);

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
        existingMove->visits += r.visits;
        existingMove->wins += r.wins;
        if (bestMove == nullptr || bestMove->successRate() < existingMove->successRate())
        {
            bestMove = existingMove;
        }
    }
    _result = *bestMove;
}

void MonteCarloScheduler::doBenchmarkDispatch(int /* node */)
{
    doDispatch();
}    
