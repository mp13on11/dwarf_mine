#include "CudaMonteCarloElf.h"
#include <OthelloField.h>
#include "MonteCarloTreeSearch.h"
#include <cuda-utils/Memory.h>
#include <iostream>
#include <random>

using namespace std;

size_t NUMBER_OF_BLOCKS = 64;
size_t MAXIMAL_NUMBER_OF_MOVES = 120;

void initialize(const OthelloState& state, vector<Field>& aggregatedPlayfields, vector<OthelloResult>& aggregatedResults)
{
    MoveList untriedMoves = state.getPossibleMoves();
    size_t size = state.playfieldSideLength() * state.playfieldSideLength();
    for (size_t i = 0; i < untriedMoves.size(); ++i)
    {
        OthelloState childState(state, untriedMoves[i]);

        const Field* buffer = childState.playfieldBuffer();
        for (size_t j = 0; j < size; ++j)
        {
            aggregatedPlayfields.push_back(buffer[j]);
        }

        aggregatedResults.emplace_back(untriedMoves[i].x, untriedMoves[i].y, 0, 0);
    }
}

OthelloResult CudaMonteCarloElf::getBestMoveFor(OthelloState& state, size_t reiterations, size_t nodeId, size_t commonSeed)
{
    vector<Field> aggregatedChildStatePlayfields;
    vector<OthelloResult> aggregatedChildResults;

    initialize(state, aggregatedChildStatePlayfields, aggregatedChildResults);

	mt19937 engine(OthelloHelper::generateUniqueSeed(nodeId, 0, commonSeed));
	uniform_real_distribution<float> generator(0, 1);
    vector<float> randomValues;	
	// #reiterations * #moves 
	// #moves + #reiterations + #blocks  = 120 + reiterations per block * blocks
	// #moves * blocks + #reiterations per block
	// 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9
	// x       x       x       x       x   
	//   x       x       x       x       x
	//     x       x       x       x       x
	//       x       x       x       x       x 
	size_t numberOfRandomValues = (MAXIMAL_NUMBER_OF_MOVES + 1) + (reiterations / NUMBER_OF_BLOCKS + 1) * NUMBER_OF_BLOCKS;
	for (size_t i = 0; i < numberOfRandomValues; ++i)
	{
		randomValues.push_back(generator(engine));
	}
	CudaUtils::Memory<float> cudaRandomValues(randomValues.size());

    //vector<size_t> seeds;
    //for (size_t i = 0; i < NUMBER_OF_BLOCKS; ++i)
    //{
    //    seeds.push_back(OthelloHelper::generateUniqueSeed(nodeId, i, commonSeed));
    //}

    //CudaUtils::Memory<size_t> cudaSeeds(seeds.size());
    CudaUtils::Memory<Field> cudaPlayfields(aggregatedChildStatePlayfields.size());
    CudaUtils::Memory<OthelloResult> cudaResults(aggregatedChildResults.size());

	cudaRandomValues.transferFrom(randomValues.data());
    //cudaSeeds.transferFrom(seeds.data());
    cudaPlayfields.transferFrom(aggregatedChildStatePlayfields.data());
    cudaResults.transferFrom(aggregatedChildResults.data());

    gameSimulationPreRandom(NUMBER_OF_BLOCKS, reiterations, cudaRandomValues.get(), aggregatedChildResults.size(), cudaPlayfields.get(), state.getCurrentEnemy(), cudaResults.get());

    cudaResults.transferTo(aggregatedChildResults.data());

    // invert results since they are calculated for the enemy player
    OthelloResult worstEnemyResult;

    for (auto& result : aggregatedChildResults)
    {
        // inverted since we calculated the successrate for the enemy
        if (worstEnemyResult.visits == 0 || 
            worstEnemyResult.successRate() >= result.successRate())
        {
            worstEnemyResult = result;
        }   
    }

    OthelloResult result = OthelloResult { 
        worstEnemyResult.x,
        worstEnemyResult.y,
        worstEnemyResult.visits,
        worstEnemyResult.visits - worstEnemyResult.wins
    };
    return result;
}

