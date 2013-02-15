#include "CudaMonteCarloElf.h"
#include <OthelloField.h>
#include "MonteCarloTreeSearch.h"
#include <cuda-utils/Memory.h>
#include <iostream>

using namespace std;

OthelloResult CudaMonteCarloElf::getBestMoveFor(OthelloState& state, size_t reiterations)
{
    MoveList untriedMoves = state.getPossibleMoves();
    vector<Field> aggregatedChildStatePlayfields;
    vector<OthelloResult> aggregatedChildResults(untriedMoves.size());
    for (size_t i = 0; i < untriedMoves.size(); ++i)
    {
        auto childState = state;
        childState.doMove(untriedMoves[i]);
        Playfield childPlayfield;

        for (int row = 0; row <  childState.playfieldSideLength(); ++row)
            for (int column = 0; column < childState.playfieldSideLength(); ++column)
                aggregatedChildStatePlayfields.push_back(childState.playfield(column, row));    
        aggregatedChildResults[i].x = 0;
        aggregatedChildResults[i].y = 0;
        aggregatedChildResults[i].wins = 0;
        aggregatedChildResults[i].visits = 0;
        aggregatedChildResults[i].iterations = 0;
    }

    

    CudaUtils::Memory<Field> cudaPlayfields(aggregatedChildStatePlayfields.size());
    CudaUtils::Memory<OthelloResult> cudaResults(aggregatedChildResults.size());

    cudaPlayfields.transferFrom(aggregatedChildStatePlayfields.data());
    cudaResults.transferFrom(aggregatedChildResults.data());

    gameSimulation(reiterations, untriedMoves.size(), cudaPlayfields.get(), state.getCurrentEnemy(), cudaResults.get());

    cudaResults.transferTo(aggregatedChildResults.data());

    // invert results since they are calculated for the enemy player

    OthelloResult worstEnemyResult;
    size_t iterations = 0;

    for (auto& result : aggregatedChildResults)
    {
        // inverted since we calculated the successrate for the enemy
        if (worstEnemyResult.visits == 0 || 
            worstEnemyResult.successRate() >= result.successRate())
        {
            worstEnemyResult = result;
        }   
        iterations += result.iterations;
    }

    OthelloResult result = OthelloResult { 
        worstEnemyResult.x,
        worstEnemyResult.y,
        worstEnemyResult.visits,
        worstEnemyResult.visits - worstEnemyResult.wins,
        iterations
    };
    return result;
}
