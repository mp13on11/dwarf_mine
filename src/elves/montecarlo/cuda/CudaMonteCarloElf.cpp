#include "CudaMonteCarloElf.h"
#include <OthelloField.h>
#include "MonteCarloTreeSearch.h"
#include <cuda-utils/Memory.h>
#include <iostream>

using namespace std;

OthelloResult CudaMonteCarloElf::getBestMoveFor(OthelloState& state, size_t reiterations)
{
    vector<Field> playfield = {
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

    compute(reiterations, state.playfieldSideLength(), cudaPlayfield.get(), cudaMoveX.get(), cudaMoveY.get(), cudaWins.get(), cudaVisits.get());

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
}
