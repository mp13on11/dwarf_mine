#include "CudaMonteCarloElf.h"
#include <OthelloField.h>
#include "MonteCarloTreeSearch.h"
#include <cuda-utils/Memory.h>
#include <iostream>
#include <cmath>
#include <cstring>
#include <cuda-utils/ErrorHandling.h>
#include <future>

using namespace std;

size_t NUMBER_OF_STREAMS = 16;
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

OthelloResult CudaMonteCarloElf::getBestMoveFor(OthelloState& state, size_t reiterations, size_t nodeId, size_t commonSeed)
{
    //return getBestMoveForSingleStream(state, reiterations, nodeId, commonSeed);
    return getBestMoveForMultipleStream(state, reiterations, nodeId, commonSeed);
}

OthelloResult CudaMonteCarloElf::getBestMoveForMultipleStream(OthelloState& state, size_t reiterations, size_t nodeId, size_t commonSeed)
{
    vector<Field> childPlayfields;
    vector<OthelloResult> childResults;
    initialize(state, childPlayfields, childResults);

    CudaUtils::Memory<Field> cudaPlayfields(childPlayfields.size());
    cudaPlayfields.transferFrom(childPlayfields.data());
    //cudaDeviceReset();
    vector<OthelloResult> collectedChildResults;
    vector<future<vector<OthelloResult>>> streamResults;

    for (size_t currentStreamId = 0; currentStreamId < NUMBER_OF_STREAMS; ++currentStreamId)
    {
        streamResults.push_back(async(launch::async, [&state, &reiterations, &nodeId, &commonSeed, &childResults, &cudaPlayfields, currentStreamId]() -> vector<OthelloResult> {
            cudaStream_t stream;
            CudaUtils::checkError(cudaStreamCreate(&stream));
          
            size_t* cudaSeeds;
            CudaUtils::checkError(cudaMalloc(&cudaSeeds, sizeof(size_t) * NUMBER_OF_BLOCKS));

            OthelloResult *cudaResults;
            CudaUtils::checkError(cudaMalloc(&cudaResults, sizeof(OthelloResult) * childResults.size()));

            size_t *hostSeeds;
            CudaUtils::checkError(cudaMallocHost(&hostSeeds, sizeof(size_t) * NUMBER_OF_BLOCKS));
            for (size_t i = 0; i < NUMBER_OF_BLOCKS; ++i)
            {
                hostSeeds[i] = OthelloHelper::generateUniqueSeed(nodeId, NUMBER_OF_STREAMS * NUMBER_OF_BLOCKS * i + currentStreamId , commonSeed);
            }

            CudaUtils::checkError(cudaMemcpyAsync(cudaSeeds, hostSeeds, sizeof(size_t) * NUMBER_OF_BLOCKS, cudaMemcpyHostToDevice, stream));
            
            OthelloResult *hostResults;
            CudaUtils::checkError(cudaMallocHost(&hostResults, sizeof(OthelloResult) * childResults.size()));
            // clean pinned memory
            copy(childResults.data(), childResults.data() + childResults.size(), hostResults);
            
            CudaUtils::checkError(cudaMemcpyAsync(cudaResults, hostResults, sizeof(OthelloResult) * childResults.size(), cudaMemcpyHostToDevice, stream));

            gameSimulationStreamed(NUMBER_OF_BLOCKS, size_t(ceil(reiterations * 1.0 / NUMBER_OF_STREAMS)), cudaSeeds, childResults.size(), cudaPlayfields.get(), state.getCurrentEnemy(), cudaResults, stream);

            CudaUtils::checkError(cudaMemcpyAsync(hostResults, cudaResults, sizeof(OthelloResult) * childResults.size(), cudaMemcpyDeviceToHost, stream));
            
            CudaUtils::checkError(cudaStreamSynchronize(stream));

            vector<OthelloResult> result(childResults.size());

            copy(hostResults, hostResults + result.size(), result.data());

            cudaFreeHost(hostResults);
            cudaFreeHost(hostSeeds);
            cudaFree(cudaSeeds);
            cudaFree(cudaResults);
            return result;
        }));
    }
    cudaDeviceSynchronize();
    for (auto& future : streamResults)
    {
        for (const auto& result : future.get())
        {
            collectedChildResults.push_back(result);
        }
    }

    vector<OthelloResult> aggregatedChildResults;
    for (auto& result: collectedChildResults)
    {
        //cout << "Stream {" << result.x << ", "<<result.y<<"}: "<<result.wins<<"/"<<result.visits<<endl;
        bool existed = false;
        for (auto& aggregatedResult : aggregatedChildResults)
        {
            if (aggregatedResult.x == result.x && aggregatedResult.y == result.y)
            {
                aggregatedResult.visits += result.visits;
                aggregatedResult.wins += result.wins;
                existed = true;
                break;
            }
        }
        if (!existed)
        {
            aggregatedChildResults.push_back(result);
        }
    }
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

OthelloResult CudaMonteCarloElf::getBestMoveForSingleStream(OthelloState& state, size_t reiterations, size_t nodeId, size_t commonSeed)
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
