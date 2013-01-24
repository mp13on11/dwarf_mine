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
#include <cmath>

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
    if (MpiHelper::isMaster(rank))
    {
        distributeInput();
        if (_repetitions != 0)
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
    
    if (nodeSet.size() == 1)
    {
        int slaveRank = (*nodeSet.begin()).first;
        if (MpiHelper::isMaster(rank))
        {
            if (slaveRank == rank)
            {
                results[0] = _result;
            }
            else
            {
                cout <<"\t"<< rank  << " receiving from "<<slaveRank<<endl;
                MPI::COMM_WORLD.Recv(results.data(), 1, MPI_OthelloResult, slaveRank, 0);
                cout <<"\t"<< rank  << " received from "<<slaveRank<<endl;
            }
        }
        else
        {
            cout <<"\t"<< rank  << " sending"<<endl;
            MPI::COMM_WORLD.Send(&_result, 1, MPI_OthelloResult, MpiHelper::MASTER, 0);
            cout <<"\t"<< rank  << " send"<<endl;
        }
    }
    else
    {
        MPI::COMM_WORLD.Gather(
            &_result, 1, MPI_OthelloResult, 
            results.data(), 1, MPI_OthelloResult, MpiHelper::MASTER);
    }
    
    return results;
}

void MonteCarloScheduler::collectInput()
{
    size_t parameters[] = {0, 0};
    MPI::COMM_WORLD.Recv(parameters, 2, MPI::UNSIGNED, MpiHelper::MASTER, 0);
    cout << "Received on "<<rank << " " << parameters[0] << " with " << parameters[1] <<" repetitions"<< endl;
    size_t bufferSize = parameters[0];
    _repetitions = parameters[1];

    Playfield playfield(bufferSize);

    auto MPI_FIELD = MPI::INT;
    MPI::COMM_WORLD.Bcast(playfield.data(), bufferSize, MPI_FIELD, MpiHelper::MASTER);
    _state = OthelloState(playfield, Player::White);
}

void MonteCarloScheduler::distributeInput()
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
            size_t parameters[] = { bufferSize, (size_t)round(_repetitions * rating.second / ratingSum) };
            MPI::COMM_WORLD.Send(parameters, 2, MPI::UNSIGNED, rating.first, 0);
            cout << "Send to "<<rating.first << " " << parameters[0] << " with " << parameters[1] <<" repetitions"<< endl;
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
    else
    {
        _repetitions = 0;
    }

    Playfield playfield(bufferSize);
    playfield.assign(_state.playfieldBuffer(), _state.playfieldBuffer() + bufferSize);
    
    cout << "Broadcast on " << rank << endl;
    auto MPI_FIELD = MPI::INT;
    MPI::COMM_WORLD.Bcast(playfield.data(), bufferSize, MPI_FIELD, MpiHelper::MASTER);
    _state = OthelloState(playfield, Player::White);
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

