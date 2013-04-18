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

#include "SMPOthelloElf.h"
#include "OthelloUtil.h"
#include <functional>
#include <random>
#include <iostream>
#include <omp.h>
#include <chrono>
#include <cassert>
#include <ratio>
#include <vector>
#include <atomic>
#include <algorithm>

using namespace std;

const int WORKSTEALING_BLOCKSIZE = 10;

void expand(const State& rootState, vector<State>& childStates, vector<Result>& childResults)
{
    auto moves = rootState.getPossibleMoves();
    for (const auto& move : moves)
    {
        childStates.emplace_back(rootState, move);
        childResults.emplace_back(Result{ (size_t)move.x, (size_t)move.y, 0, 0});
    }   
}

void rollout(State& state, RandomGenerator generator)
{
    size_t passCounter = 0;
    while (passCounter < 2)
    {
        auto possibleMoves = state.getPossibleMoves();
        if (possibleMoves.size() > 0)
        {
            Move move = possibleMoves[generator(possibleMoves.size())];
            state.doMove(move);
            passCounter = 0;
        }
        else
        {
            ++passCounter;
        }
    }
}

vector<Result> SMPOthelloElf::getMovesFor(State& rootState, size_t reiterations, size_t nodeId, size_t commonSeed)
{
    vector<Result> childResults;
    vector<State> childStates;

    expand(rootState, childStates, childResults);

    #pragma omp parallel
    {
        // thread-private random number generator
        mt19937 engine(OthelloHelper::generateUniqueSeed(nodeId, omp_get_thread_num(), commonSeed));

        auto generator = [&engine](size_t limit){ 
            uniform_int_distribution<size_t> distribution(0, limit - 1);
            return distribution(engine);
        };

        #pragma omp for schedule(dynamic, WORKSTEALING_BLOCKSIZE)
        for(size_t i=0; i<reiterations; ++i)
        {
            // select randomly
            size_t selectedIndex = generator(childStates.size());
            State selectedState = childStates[selectedIndex];

            // rollout
            rollout(selectedState, generator);

            // backpropagate
            #pragma omp critical(selectedIndex)
            {
                childResults[selectedIndex].visits++;
                if (selectedState.hasWon(rootState.getCurrentPlayer()))
                    childResults[selectedIndex].wins++;
            }
        }
    }

    return childResults;
