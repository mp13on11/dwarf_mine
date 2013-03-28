#include "CudaOthelloElf.h"
#include <Field.h>
#include "CudaProxy.h"
#include <cuda-utils/Memory.h>
#include <iostream>
#include <random>
#include <future>
#include "common/Utils.h"


using namespace std;

size_t NUMBER_OF_STREAMS = 2;
size_t NUMBER_OF_BLOCKS = 1;
size_t MAXIMAL_NUMBER_OF_MOVES = 120;

void initialize(const State& state, vector<Field>& aggregatedPlayfields, vector<Result>& aggregatedResults)
{
    MoveList untriedMoves = state.getPossibleMoves();
    size_t size = state.playfieldSideLength() * state.playfieldSideLength();
    for (size_t i = 0; i < untriedMoves.size(); ++i)
    {
        State childState(state, untriedMoves[i]);

        const Field* buffer = childState.playfieldBuffer();
        for (size_t j = 0; j < size; ++j)
        {
            aggregatedPlayfields.push_back(buffer[j]);
        }

        aggregatedResults.emplace_back(untriedMoves[i].x, untriedMoves[i].y, 0, 0);
    }
}

Result CudaOthelloElf::getBestMoveForStreamed(State& state, size_t reiterations, size_t nodeId, size_t commonSeed)
{
    vector<Field> childPlayfields;
    vector<Result> childResults;
    initialize(state, childPlayfields, childResults);

    CudaUtils::Memory<Field> cudaPlayfields(childPlayfields.size());
    cudaPlayfields.transferFrom(childPlayfields.data());
    //cudaDeviceReset();
    vector<Result> collectedChildResults;
    vector<future<vector<Result>>> streamResults;

    size_t reiterationsPerStream = div_ceil(reiterations,NUMBER_OF_STREAMS);
    //size_t numberOfRandomValues = (MAXIMAL_NUMBER_OF_MOVES + 1) + (reiterations / NUMBER_OF_BLOCKS + 1) * NUMBER_OF_BLOCKS;

    for (size_t currentStreamId = 0; currentStreamId < NUMBER_OF_STREAMS; ++currentStreamId)
    {
        streamResults.push_back(async(launch::async, [&state, &reiterationsPerStream, &nodeId, &commonSeed, &childResults, &cudaPlayfields, currentStreamId]() -> vector<Result> {
            cudaStream_t stream;
            CudaUtils::checkError(cudaStreamCreate(&stream));

            size_t* cudaSeeds;
            CudaUtils::checkError(cudaMalloc(&cudaSeeds, sizeof(size_t) * NUMBER_OF_BLOCKS));

            Result *cudaResults;
            CudaUtils::checkError(cudaMalloc(&cudaResults, sizeof(Result) * childResults.size()));

            size_t numberOfRandomValues = (MAXIMAL_NUMBER_OF_MOVES + 1) + div_ceil(reiterationsPerStream, NUMBER_OF_BLOCKS) * NUMBER_OF_BLOCKS;

            CudaUtils::Memory<float> randomValues(numberOfRandomValues);

            Result *hostResults;
            CudaUtils::checkError(cudaMallocHost(&hostResults, sizeof(Result) * childResults.size()));
            // clean pinned memory
            copy(childResults.data(), childResults.data() + childResults.size(), hostResults);

            CudaUtils::checkError(cudaMemcpyAsync(cudaResults, hostResults, sizeof(Result) * childResults.size(), cudaMemcpyHostToDevice, stream));
			size_t streamSeed = OthelloHelper::generateUniqueSeed(nodeId, (size_t)stream, commonSeed);
            gameSimulationPreRandom(NUMBER_OF_BLOCKS, reiterationsPerStream, randomValues.get(), numberOfRandomValues, childResults.size(), cudaPlayfields.get(), state.getCurrentEnemy(), cudaResults, stream, streamSeed);

            CudaUtils::checkError(cudaMemcpyAsync(hostResults, cudaResults, sizeof(Result) * childResults.size(), cudaMemcpyDeviceToHost, stream));
            
            CudaUtils::checkError(cudaStreamSynchronize(stream));

            vector<Result> result(childResults.size());

            copy(hostResults, hostResults + result.size(), result.data());

            cudaFreeHost(hostResults);
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

    vector<Result> aggregatedChildResults;
    for (auto& result: collectedChildResults)
    {
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
    Result worstEnemyResult;

    for (auto& result : aggregatedChildResults)
    {
        // inverted since we calculated the successrate for the enemy
        if (worstEnemyResult.visits == 0 || 
            worstEnemyResult.successRate() >= result.successRate())
        {
            worstEnemyResult = result;
        }   
    }

    Result result = Result { 
        worstEnemyResult.x,
        worstEnemyResult.y,
        worstEnemyResult.visits,
        worstEnemyResult.visits - worstEnemyResult.wins
    };
    return result;
}

Result CudaOthelloElf::getBestMoveFor(State& state, size_t reiterations, size_t nodeId, size_t commonSeed)
{
    //return getBestMoveForSimple(state, reiterations, nodeId, commonSeed);
    return getBestMoveForStreamed(state, reiterations, nodeId, commonSeed);
}

Result CudaOthelloElf::getBestMoveForSimple(State& state, size_t reiterations, size_t /*nodeId */, size_t /*commonSeed*/)
{
    vector<Field> aggregatedChildStatePlayfields;
    vector<Result> aggregatedChildResults;

    initialize(state, aggregatedChildStatePlayfields, aggregatedChildResults);

    // one iteration: max 120 moves + one selection for leaf - for reiteration variation we step for each iteration one index to the right
	size_t numberOfRandomValues = (MAXIMAL_NUMBER_OF_MOVES + 1) + (reiterations / NUMBER_OF_BLOCKS + 1) * NUMBER_OF_BLOCKS;
    vector<float> randomValues(numberOfRandomValues);
	CudaUtils::Memory<float> cudaRandomValues(randomValues.size());

    CudaUtils::Memory<Field> cudaPlayfields(aggregatedChildStatePlayfields.size());
    CudaUtils::Memory<Result> cudaResults(aggregatedChildResults.size());

	cudaRandomValues.transferFrom(randomValues.data());
    cudaPlayfields.transferFrom(aggregatedChildStatePlayfields.data());
    cudaResults.transferFrom(aggregatedChildResults.data());

    gameSimulationPreRandom(NUMBER_OF_BLOCKS, reiterations, cudaRandomValues.get(), cudaRandomValues.numberOfElements(), aggregatedChildResults.size(), cudaPlayfields.get(), state.getCurrentEnemy(), cudaResults.get());

    cudaResults.transferTo(aggregatedChildResults.data());

    // invert results since they are calculated for the enemy player
    Result worstEnemyResult;

    for (auto& result : aggregatedChildResults)
    {
        // inverted since we calculated the successrate for the enemy
        if (worstEnemyResult.visits == 0 || 
            worstEnemyResult.successRate() >= result.successRate())
        {
            worstEnemyResult = result;
        }   
    }

    Result result = Result { 
        worstEnemyResult.x,
        worstEnemyResult.y,
        worstEnemyResult.visits,
        worstEnemyResult.visits - worstEnemyResult.wins
    };
    return result;
}

