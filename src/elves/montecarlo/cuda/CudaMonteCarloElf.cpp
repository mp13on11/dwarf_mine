#include "CudaMonteCarloElf.h"
#include <OthelloField.h>
#include "MonteCarloTreeSearch.h"
#include <cuda-utils/Memory.h>
#include <iostream>
#include <cmath>
#include <cstring>
#include <cuda-utils/ErrorHandling.h>

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

OthelloResult CudaMonteCarloElf::getBestMoveFor(OthelloState& state, size_t reiterations, size_t nodeId, size_t commonSeed)
{
    //return getBestMoveForSingleStream(state, reiterations, nodeId, commonSeed);
    return getBestMoveForMultipleStream(state, reiterations, nodeId, commonSeed);
}

OthelloResult CudaMonteCarloElf::getBestMoveForMultipleStream(OthelloState& state, size_t reiterations, size_t nodeId, size_t commonSeed)
{
    cudaStream_t stream1, stream2;
    CudaUtils::checkError(cudaStreamCreate(&stream1));
    CudaUtils::checkError(cudaStreamCreate(&stream2));

    vector<Field> aggregatedChildStatePlayfields1;
    vector<Field> aggregatedChildStatePlayfields2;
    vector<OthelloResult> aggregatedChildResults1;
    vector<OthelloResult> aggregatedChildResults2;
    initialize(state, aggregatedChildStatePlayfields1, aggregatedChildResults1);
    initialize(state, aggregatedChildStatePlayfields2, aggregatedChildResults2);

    vector<size_t> seeds1;
    vector<size_t> seeds2;
    for (size_t i = 0; i < NUMBER_OF_BLOCKS; ++i)
    {
        seeds1.push_back(OthelloHelper::generateUniqueSeed(nodeId, pow(2, i), commonSeed));
        seeds2.push_back(OthelloHelper::generateUniqueSeed(nodeId, pow(2, i) + 1, commonSeed));
    }
    //CudaUtils::StreamedMemory<size_t> cudaSeeds1(seeds1.size());
    //CudaUtils::StreamedMemory<size_t> cudaSeeds2(seeds2.size());
    size_t *cudaSeeds1, *cudaSeeds2;
    CudaUtils::checkError(cudaMalloc(&cudaSeeds1, sizeof(size_t) * seeds1.size()));
    CudaUtils::checkError(cudaMalloc(&cudaSeeds2, sizeof(size_t) * seeds2.size()));

    //CudaUtils::StreamedMemory<Field> cudaPlayfields1(aggregatedChildStatePlayfields1.size());
    //CudaUtils::StreamedMemory<Field> cudaPlayfields2(aggregatedChildStatePlayfields2.size());
    Field *cudaPlayfields1, *cudaPlayfields2;
    CudaUtils::checkError(cudaMalloc(&cudaPlayfields1, sizeof(Field) * aggregatedChildStatePlayfields1.size()));
    CudaUtils::checkError(cudaMalloc(&cudaPlayfields2, sizeof(Field) * aggregatedChildStatePlayfields2.size()));
    // CudaUtils::StreamedMemory<OthelloResult> cudaResults1(aggregatedChildResults1.size());
    // CudaUtils::StreamedMemory<OthelloResult> cudaResults2(aggregatedChildResults2.size());
    OthelloResult *cudaResults1, *cudaResults2;
    CudaUtils::checkError(cudaMalloc(&cudaResults1, sizeof(OthelloResult) * aggregatedChildResults1.size()));
    CudaUtils::checkError(cudaMalloc(&cudaResults2, sizeof(OthelloResult) * aggregatedChildResults2.size()));

    // cudaSeeds1.transferFrom(seeds1.data());
    // cudaSeeds2.transferFrom(seeds2.data());
    size_t *hostSeeds1, *hostSeeds2;
    CudaUtils::checkError(cudaMallocHost(&hostSeeds1, sizeof(size_t) * seeds1.size()));
    CudaUtils::checkError(cudaMallocHost(&hostSeeds2, sizeof(size_t) * seeds2.size()));
    copy(seeds1.data(), seeds1.data() + seeds1.size(), hostSeeds1);
    copy(seeds2.data(), seeds2.data() + seeds2.size(), hostSeeds2);
    CudaUtils::checkError(cudaMemcpyAsync(cudaSeeds1, hostSeeds1, sizeof(size_t) * seeds1.size(), cudaMemcpyHostToDevice, stream1));
    CudaUtils::checkError(cudaMemcpyAsync(cudaSeeds2, hostSeeds2, sizeof(size_t) * seeds2.size(), cudaMemcpyHostToDevice, stream2));
    // cudaPlayfields1.transferFrom(aggregatedChildStatePlayfields1.data());
    // cudaPlayfields2.transferFrom(aggregatedChildStatePlayfields2.data());

    Field *hostPlayfields1, *hostPlayfields2;
    CudaUtils::checkError(cudaMallocHost(&hostPlayfields1, sizeof(Field) * aggregatedChildStatePlayfields1.size()));
    CudaUtils::checkError(cudaMallocHost(&hostPlayfields2, sizeof(Field) * aggregatedChildStatePlayfields2.size()));
    copy(aggregatedChildStatePlayfields1.data(), aggregatedChildStatePlayfields1.data() + aggregatedChildStatePlayfields1.size(), hostPlayfields1);
    copy(aggregatedChildStatePlayfields2.data(), aggregatedChildStatePlayfields2.data() + aggregatedChildStatePlayfields2.size(), hostPlayfields2);
    CudaUtils::checkError(cudaMemcpyAsync(cudaPlayfields1, hostPlayfields1, sizeof(Field) * aggregatedChildStatePlayfields1.size(), cudaMemcpyHostToDevice, stream1));
    CudaUtils::checkError(cudaMemcpyAsync(cudaPlayfields2, hostPlayfields2, sizeof(Field) * aggregatedChildStatePlayfields2.size(), cudaMemcpyHostToDevice, stream2));
    // cudaResults1.transferFrom(aggregatedChildResults1.data());
    // cudaResults2.transferFrom(aggregatedChildResults2.data());

    OthelloResult *hostResults1, *hostResults2;
    CudaUtils::checkError(cudaMallocHost(&hostResults1, sizeof(OthelloResult) * aggregatedChildResults1.size()));
    CudaUtils::checkError(cudaMallocHost(&hostResults2, sizeof(OthelloResult) * aggregatedChildResults2.size()));
    copy(aggregatedChildResults1.data(), aggregatedChildResults1.data() + aggregatedChildResults1.size(), hostResults1);
    copy(aggregatedChildResults2.data(), aggregatedChildResults2.data() + aggregatedChildResults2.size(), hostResults2);
    CudaUtils::checkError(cudaMemcpyAsync(cudaResults1, hostResults1, sizeof(OthelloResult) * aggregatedChildResults1.size(), cudaMemcpyHostToDevice, stream1));
    CudaUtils::checkError(cudaMemcpyAsync(cudaResults2, hostResults2, sizeof(OthelloResult) * aggregatedChildResults2.size(), cudaMemcpyHostToDevice, stream2));
    
    gameSimulationStreamed(NUMBER_OF_BLOCKS, size_t(ceil(reiterations / 2.0)), cudaSeeds1, aggregatedChildResults1.size(), cudaPlayfields1, state.getCurrentEnemy(), cudaResults1, stream1);
    gameSimulationStreamed(NUMBER_OF_BLOCKS, size_t(ceil(reiterations / 2.0)), cudaSeeds2, aggregatedChildResults2.size(), cudaPlayfields2, state.getCurrentEnemy(), cudaResults2, stream2);

    // cudaResults1.transferTo(aggregatedChildResults1.data());
    // cudaResults2.transferTo(aggregatedChildResults2.data());
    
    CudaUtils::checkError(cudaStreamSynchronize(stream1));
    CudaUtils::checkError(cudaStreamSynchronize(stream2));
    
    cudaMemcpyAsync(hostResults1, cudaResults1, sizeof(OthelloResult) * aggregatedChildResults1.size(), cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(hostResults2, cudaResults2, sizeof(OthelloResult) * aggregatedChildResults2.size(), cudaMemcpyDeviceToHost, stream2);
    
    cudaDeviceSynchronize();
    CudaUtils::checkState();
    copy(hostResults1, hostResults1 + aggregatedChildResults1.size(), aggregatedChildResults1.data());
    copy(hostResults2, hostResults2 + aggregatedChildResults2.size(), aggregatedChildResults2.data());


    // CudaUtils::checkError(cudaMemcpy(cudaResults1, aggregatedChildResults1.data(), sizeof(OthelloResult) * aggregatedChildResults1.size(), cudaMemcpyDeviceToHost));
    // CudaUtils::checkError(cudaMemcpy(cudaResults2, aggregatedChildResults2.data(), sizeof(OthelloResult) * aggregatedChildResults2.size(), cudaMemcpyDeviceToHost));
    // invert results since they are calculated for the enemy player
    OthelloResult worstEnemyResult;

    vector<OthelloResult> aggregatedChildResults;
    for (auto& result: aggregatedChildResults1)
    {
        aggregatedChildResults.push_back(result);
        //cout << "Stream 1 {" << result.x << ", "<<result.y<<"}: "<<result.wins<<"/"<<result.visits<<endl;
    }
    for (auto& result: aggregatedChildResults2)
    {
        //cout << "Stream 2 {" << result.x << ", "<<result.y<<"}: "<<result.wins<<"/"<<result.visits<<endl;
        for (auto& aggregatedResult : aggregatedChildResults)
        {
            if (aggregatedResult.x == result.x && aggregatedResult.y == result.y)
            {
                aggregatedResult.visits += result.visits;
                aggregatedResult.wins += result.wins;
            }
            break;
        }
    }

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
