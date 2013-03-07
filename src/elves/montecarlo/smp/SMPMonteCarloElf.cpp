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
#include <atomic>

using namespace std;

const int WORKSTEALING_BLOCKSIZE = 10;

void expand(const OthelloState& rootState, vector<OthelloState>& childStates, vector<OthelloResult>& childResults)
{
    auto moves = rootState.getPossibleMoves();
    for (const auto& move : moves)
    {
        childStates.emplace_back(rootState, move);
        childResults.emplace_back(OthelloResult{ (size_t)move.x, (size_t)move.y, 0, 0});
    }   
}

void rollout(OthelloState& state, RandomGenerator generator)
{
    size_t passCounter = 0;
    while (passCounter < 2)
    {
        auto possibleMoves = state.getPossibleMoves();
        if (possibleMoves.size() > 0)
        {
            OthelloMove move = possibleMoves[generator(possibleMoves.size())];
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

    size_t executedIterations = 0;
    // no parallel for to enable a form of workstealing 
    // threads do not have to execute the same number of cycles but instead 
    #pragma omp parallel shared(executedIterations, childStates, childResults) 
    {
        while (executedIterations < reiterations)
        {
            size_t start = 0;
            size_t end = 0;
            // calculate the next block to iterate over
            #pragma omp critical
            {
                start = executedIterations;
                end = min(start + WORKSTEALING_BLOCKSIZE, reiterations);
                executedIterations += end - start;
            }
            for (size_t i = start; i < end; ++i)
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
                    if (selectedState.hasWon(selectedState.getCurrentEnemy()))
                        childResults[selectedIndex].wins++;
                }
            }
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
    return *bestResult;
}