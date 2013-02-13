#include "CudaMonteCarloElf.h"
#include <OthelloField.h>
#include "MonteCarloTreeSearch.h"
#include <cuda-utils/Memory.h>
#include <iostream>

using namespace std;

OthelloResult CudaMonteCarloElf::getBestMoveFor(OthelloState& state, size_t reiterations)
{
    OthelloState debugState(
        {W, W, W, W, W, W, W, W,
         W, W, W, W, W, W, W, W,
         W, W, W, W, B, B, B, B,
         W, W, W, W, B, B, B, B,      
         W, W, W, W, B, B, B, B,      
         W, W, W, W, W, W, F, F,
         W, W, W, W, F, F, F, F,
         W, W, W, W, F, F, F, F}, Player::Black);
//state = debugState;
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
    }

    

    CudaUtils::Memory<Field> cudaPlayfields(aggregatedChildStatePlayfields.size());
    CudaUtils::Memory<OthelloResult> cudaResults(aggregatedChildResults.size());

    cudaPlayfields.transferFrom(aggregatedChildStatePlayfields.data());
    cudaResults.transferFrom(aggregatedChildResults.data());

    gameSimulation(reiterations, untriedMoves.size(), cudaPlayfields.get(), state.getCurrentEnemy(), cudaResults.get());

    cudaResults.transferTo(aggregatedChildResults.data());

    OthelloResult bestResult;
    size_t iterations = 0;
    int i = 0;
    for (auto& result : aggregatedChildResults)
    {
        cout << i++ << "\n"<<"\twins:\t"<<result.wins<<"\n\tvisits:\t"<<result.visits<<"\n\tmoveX:\t"<<result.x<<"\n\tmoveY:\t"<<result.y<<"\n\titerations:\t"<<result.iterations<<endl;
        //OthelloHelper::writeResultToStream(cout, result);
        cout << endl;
        // inverted since we calculated the successrate for the enemy
        if (bestResult.iterations == 0 || 
            bestResult.successRate() > result.successRate())
        {
            bestResult = result;
        }   
        iterations += result.iterations;
    }
    bestResult.wins = bestResult.visits - bestResult.wins;
    bestResult.iterations = iterations;
    return bestResult;
/*
    vector<Field> playfield = {cudaResults
        F, F, F, F, F, F, F, F,
        F, F, F, F, F, F, F, F,
        F, F, F, F, F, F, F, F,
        F, F, F, W, B, F, F, F,      
        F, F, F, B, W, F, F, F,      
        F, F, F, F, F, F, F, F,
        F, F, F, F, F, F, F, F,
        F, F, F, F, F, F, F, F
    };
	size_t size = state.playfieldSideLength() * state.playfieldSideLength();
    CudaUtils::Memory<Field> cudaPlayfield(size);
    CudaUtils::Memory<size_t> cudaVisits(1);
    CudaUtils::Memory<size_t> cudaWins(1);
    CudaUtils::Memory<size_t> cudaMoveX(1);
    CudaUtils::Memory<size_t> cudaMoveY(1);    

    cudaPlayfield.transferFrom(playfield.data());

    leafSimulation(reiterations, state.playfieldSideLength(), cudaPlayfield.get(), state.getCurrentPlayer(), cudaMoveX.get(), cudaMoveY.get(), cudaWins.get(), cudaVisits.get());

    OthelloResult result;
    Playfield buffer(size);
    cudaPlayfield.transferTo(buffer.data());
    OthelloState state2(buffer, White);
    cout << state2 << endl;
    cudaMoveX.transferTo(&(result.x));
    cudaMoveY.transferTo(&(result.y));
    cudaVisits.transferTo(&(result.visits));
    cudaWins.transferTo(&(result.wins));
    result.iterations = reiterations;
    return result;
*/
}
