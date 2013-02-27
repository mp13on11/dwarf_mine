#include "SMPMonteCarloElf.h"
#include "OthelloUtil.h"
#include <functional>
#include <random>
#include <iostream>
#include <omp.h>
#include <chrono>
#include <cassert>
#include <ratio>
#include <vector>

using namespace std;

void expand(const OthelloState& rootState, vector<OthelloState>& childStates, vector<OthelloResult>& childResults)
{
    auto moves = rootState.getPossibleMoves();
    for (const auto& move : moves)
    {
        childStates.emplace_back(rootState, move);
        childResults.emplace_back(OthelloResult{ (size_t)move.x, (size_t)move.y, 0, 0, 0});
    }   
}

void rollout(OthelloState& state, RandomGenerator generator)
{
    size_t passCounter = 0;
    while (passCounter < 2)
    {
        if (state.hasPossibleMoves())
        {
            OthelloMove move = state.getRandomMove(generator);
            state.doMove(move);
            passCounter = 0;
        }
        else
        {
            ++passCounter;
        }
    }
}

OthelloResult SMPMonteCarloElf::getBestMoveFor(OthelloState& rootState, size_t reiterations, size_t nodeId, size_t commonSeed)
{
    size_t threadCount = omp_get_max_threads();
    vector<mt19937> engines;
    for (size_t i = 0; i < threadCount; ++i)
    {
        engines.emplace_back(OthelloHelper::generateUniqueSeed(nodeId, i, commonSeed));
    }

    auto generator = [&engines](size_t limit){ 
        uniform_int_distribution<size_t> distribution(0, limit - 1);
        return distribution(engines[omp_get_thread_num()]);
    };    


    vector<OthelloResult> childResults;
    vector<OthelloState> childStates;

    expand(rootState, childStates, childResults);
  
    #pragma omp parallel for shared(rootState, childStates, childResults) 
    for (size_t i = 0; i < reiterations; ++i)
    {
        // select
        size_t selectedIndex = generator(childStates.size());
        OthelloState selectedState = childStates[selectedIndex];

        // roleout
        rollout(selectedState, generator);

        // backpropagate
        #pragma omp critical(selectedIndex)
        {
            childResults[selectedIndex].visits++;
            if (selectedState.hasWon(rootState.getCurrentPlayer()))
                childResults[selectedIndex].wins++;
        }
    }

    OthelloResult* bestResult = nullptr;
    for (auto& result : childResults)
    {
        if (bestResult == nullptr || bestResult->successRate() < result.successRate())
        {
            bestResult = &result;
        }
    }
    bestResult->iterations = reiterations;
    return *bestResult;
}