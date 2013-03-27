#include "CudaMonteCarloElf.h"
#include <OthelloField.h>
#include "MonteCarloTreeSearch.h"
#include <cuda-utils/Memory.h>
#include <iostream>
#include <random>
#include <future>
#include "common/Utils.h"


using namespace std;

size_t NUMBER_OF_STREAMS = 2;
size_t NUMBER_OF_BLOCKS = 1;
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

OthelloResult CudaMonteCarloElf::getBestMoveForStreamed(OthelloState& state, size_t reiterations, size_t nodeId, size_t commonSeed)
{
    vector<Field> childPlayfields;
    vector<OthelloResult> childResults;
    initialize(state, childPlayfields, childResults);

    CudaUtils::Memory<Field> cudaPlayfields(childPlayfields.size());
    cudaPlayfields.transferFrom(childPlayfields.data());
    //cudaDeviceReset();
    vector<OthelloResult> collectedChildResults;
    vector<future<vector<OthelloResult>>> streamResults;

    size_t reiterationsPerStream = div_ceil(reiterations,NUMBER_OF_STREAMS);
    //size_t numberOfRandomValues = (MAXIMAL_NUMBER_OF_MOVES + 1) + (reiterations / NUMBER_OF_BLOCKS + 1) * NUMBER_OF_BLOCKS;

    for (size_t currentStreamId = 0; currentStreamId < NUMBER_OF_STREAMS; ++currentStreamId)
    {
        streamResults.push_back(async(launch::async, [&state, &reiterationsPerStream, &nodeId, &commonSeed, &childResults, &cudaPlayfields, currentStreamId]() -> vector<OthelloResult> {
            cudaStream_t stream;
            CudaUtils::checkError(cudaStreamCreate(&stream));

            size_t* cudaSeeds;
            CudaUtils::checkError(cudaMalloc(&cudaSeeds, sizeof(size_t) * NUMBER_OF_BLOCKS));

            OthelloResult *cudaResults;
            CudaUtils::checkError(cudaMalloc(&cudaResults, sizeof(OthelloResult) * childResults.size()));

            size_t numberOfRandomValues = (MAXIMAL_NUMBER_OF_MOVES + 1) + div_ceil(reiterationsPerStream, NUMBER_OF_BLOCKS) * NUMBER_OF_BLOCKS;

            CudaUtils::Memory<float> randomValues(numberOfRandomValues);

            OthelloResult *hostResults;
            CudaUtils::checkError(cudaMallocHost(&hostResults, sizeof(OthelloResult) * childResults.size()));
            // clean pinned memory
            copy(childResults.data(), childResults.data() + childResults.size(), hostResults);

            CudaUtils::checkError(cudaMemcpyAsync(cudaResults, hostResults, sizeof(OthelloResult) * childResults.size(), cudaMemcpyHostToDevice, stream));

            gameSimulationPreRandom(NUMBER_OF_BLOCKS, reiterationsPerStream, randomValues.get(), numberOfRandomValues, childResults.size(), cudaPlayfields.get(), state.getCurrentEnemy(), cudaResults, stream);

            CudaUtils::checkError(cudaMemcpyAsync(hostResults, cudaResults, sizeof(OthelloResult) * childResults.size(), cudaMemcpyDeviceToHost, stream));
            
            CudaUtils::checkError(cudaStreamSynchronize(stream));

            vector<OthelloResult> result(childResults.size());

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

OthelloResult CudaMonteCarloElf::getBestMoveFor(OthelloState& state, size_t reiterations, size_t nodeId, size_t commonSeed)
{
    //return getBestMoveForSimple(state, reiterations, nodeId, commonSeed);
    return getBestMoveForStreamed(state, reiterations, nodeId, commonSeed);
}

OthelloResult CudaMonteCarloElf::getBestMoveForSimple(OthelloState& state, size_t reiterations, size_t /*nodeId */, size_t /*commonSeed*/)
{
    vector<Field> aggregatedChildStatePlayfields;
    vector<OthelloResult> aggregatedChildResults;

    initialize(state, aggregatedChildStatePlayfields, aggregatedChildResults);

	// mt19937 engine(OthelloHelper::generateUniqueSeed(nodeId, 0, commonSeed));
	// uniform_real_distribution<float> generator(0, 1);
    
	// #reiterations * #moves 
	// #moves + #reiterations + #blocks  = 120 + reiterations per block * blocks
	// #moves * blocks + #reiterations per block
	// 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9
	// x       x       x       x       x   
	//   x       x       x       x       x
	//     x       x       x       x       x
	//       x       x       x       x       x 
    // one iteration: max 120 moves + one selection for leaf - for reiteration variation we step for each iteration one index to the right
	size_t numberOfRandomValues = (MAXIMAL_NUMBER_OF_MOVES + 1) + (reiterations / NUMBER_OF_BLOCKS + 1) * NUMBER_OF_BLOCKS;
    //vector<float> randomValues; 
    vector<float> randomValues(numberOfRandomValues);
	// for (size_t i = 0; i < numberOfRandomValues; ++i)
	// {
	// 	randomValues.push_back(generator(engine));
	// }
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

    gameSimulationPreRandom(NUMBER_OF_BLOCKS, reiterations, cudaRandomValues.get(), cudaRandomValues.numberOfElements(), aggregatedChildResults.size(), cudaPlayfields.get(), state.getCurrentEnemy(), cudaResults.get());

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

