#include "CudaMonteCarloElf.h"
#include <OthelloField.h>
#include "MonteCarloTreeSearch.h"
#include <cuda-utils/Memory.h>
#include <iostream>

using namespace std;

size_t NUMBER_OF_BLOCKS = 64;

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

vector<OthelloResult> CudaMonteCarloElf::getMovesFor(OthelloState& state, size_t reiterations, size_t nodeId, size_t commonSeed)
{
    vector<Field> aggregatedChildStatePlayfields;
    vector<OthelloResult> aggregatedChildResults;

    initialize(state, aggregatedChildStatePlayfields, aggregatedChildResults);

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

    gameSimulation(NUMBER_OF_BLOCKS, reiterations, cudaSeeds.get(), aggregatedChildResults.size(), cudaPlayfields.get(), state.getCurrentEnemy(), cudaResults.get());

    cudaResults.transferTo(aggregatedChildResults.data());

    return aggregatedChildResults;
}

