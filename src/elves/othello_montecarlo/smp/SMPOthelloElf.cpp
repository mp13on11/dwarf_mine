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

Result SMPOthelloElf::getBestMoveFor(State& rootState, size_t reiterations, size_t nodeId, size_t commonSeed)
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
                if (selectedState.hasWon(selectedState.getCurrentEnemy()))
                    childResults[selectedIndex].wins++;
            }
        }
    }

    return *max_element(childResults.begin(), childResults.end());
}