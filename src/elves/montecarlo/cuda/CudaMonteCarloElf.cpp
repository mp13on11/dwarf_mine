#include "CudaMonteCarloElf.h"
#include <OthelloField.h>
#include "MonteCarloTreeSearch.h"
#include <cuda-utils/Memory.h>
#include <iostream>

using namespace std;

const size_t NUMBER_OF_BLOCKS = 2;

OthelloResult CudaMonteCarloElf::getBestMoveFor(OthelloState& state, size_t reiterations, size_t nodeId, size_t commonSeed)
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
        aggregatedChildResults[i].x = untriedMoves[i].x;
        aggregatedChildResults[i].y = untriedMoves[i].y;
        aggregatedChildResults[i].wins = 0;
        aggregatedChildResults[i].visits = 0;
        aggregatedChildResults[i].iterations = 0;
    }

    vector<size_t> seeds;
    for (size_t i = 0; i < NUMBER_OF_BLOCKS; ++i)
    {
        seeds.push_back(OthelloHelper::generateUniqueSeed(nodeId, i, commonSeed));
    }

    CudaUtils::Memory<size_t> cudaSeeds(seeds.size());
    CudaUtils::Memory<Field> cudaPlayfields(aggregatedChildStatePlayfields.size());
    CudaUtils::Memory<OthelloResult> cudaResults(aggregatedChildResults.size());

    cudaSeeds.transferFrom(seeds.data());
    cudaPlayfields.transferFrom(aggregatedChildStatePlayfields.data());
    cudaResults.transferFrom(aggregatedChildResults.data());

    gameSimulation(NUMBER_OF_BLOCKS, reiterations, cudaSeeds.get(), untriedMoves.size(), cudaPlayfields.get(), state.getCurrentEnemy(), cudaResults.get());

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
