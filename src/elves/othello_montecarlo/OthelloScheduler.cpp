/*****************************************************************************
* Dwarf Mine - The 13-11 Benchmark
*
* Copyright (c) 2013 Bünger, Thomas; Kieschnick, Christian; Kusber,
* Michael; Lohse, Henning; Wuttke, Nikolai; Xylander, Oliver; Yao, Gary;
* Zimmermann, Florian
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*****************************************************************************/

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
    // cout<< "Move: ("<<_result.x<<", "<<_result.y<<")\n"
    //     << "Wins: "<<_result.wins<<"/"<<_result.visits<<" = " <<_result.successRate()<<endl;
    OthelloHelper::writeResultToStream(output, _result);
}

void OthelloScheduler::doSimpleDispatch()
{
    _localRepetitions = _repetitions;
    calculate();
}

void OthelloScheduler::doDispatch()
{
    distributeInput();
    calculate();
    collectResults();
}

void OthelloScheduler::calculate()
{
    if (_localRepetitions == 0)
        return;

    _results = elf().getMovesFor(_state, _localRepetitions, communicator.rank(), _commonSeed);
}

void OthelloScheduler::distributePlayfield(size_t bufferSize)
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

    _state = State(playfield, Player::White);
}

void registerResultToMPI(MPI_Datatype& type)
{
    MPI_Datatype elementTypes[] = { 
        MPI::UNSIGNED, 
        MPI::UNSIGNED, 
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
        sizeof(unsigned int),
        2 * sizeof(unsigned int),
        3 * sizeof(unsigned int)
    };

    MPI_Type_create_struct(4, elementLengths, elementDisplacements, elementTypes, &type);
    MPI_Type_commit(&type);
}

vector<Result> OthelloScheduler::gatherResults()
{
    size_t numberOfNodes = communicator.nodeSet().size();
    size_t numberOfPossibleMoves = _state.getPossibleMoves().size();
    vector<Result> results(numberOfNodes * numberOfPossibleMoves);

    MPI_Datatype MPI_Result;
    registerResultToMPI(MPI_Result);

    communicator->Gather(
        _results.data(), _results.size(), MPI_Result,
        results.data(), numberOfPossibleMoves, MPI_Result,
        Communicator::MASTER_RANK
    );
    return results;
}

void OthelloScheduler::distributeInput()
{
    size_t bufferSize = distributeCommonParameters();
    distributePlayfield(bufferSize);
}

size_t OthelloScheduler::distributeCommonParameters()
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

void OthelloScheduler::collectResults()
{
    vector<Result> results = gatherResults();

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
    _results = accumulatedResults;
    _result = *bestMove;
}

//void MonteCarloScheduler::doBenchmarkDispatch(int node )
//{
//    BenchmarkResult benchmarkNodeset;
//    benchmarkNodeset[node] = 1;
//    doDispatch(benchmarkNodeset);
//}    
