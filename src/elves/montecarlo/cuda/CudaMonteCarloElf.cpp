#include "CudaMonteCarloElf.h"
#include <OthelloField.h>
#include "MonteCarloTreeSearch.h"
#include <cuda-utils/Memory.h>
#include <iostream>

using namespace std;
OthelloResult CudaMonteCarloElf::getBestMoveFor(OthelloState& state, size_t reiterations)
{
	size_t size = state.playfieldSideLength() * state.playfieldSideLength();
    cout << state << " " << reiterations << endl;
    CudaUtils::Memory<Field> cudaPlayfield(size);
    CudaUtils::Memory<size_t> cudaVisits(1);
    CudaUtils::Memory<size_t> cudaWins(1);
    CudaUtils::Memory<size_t> cudaMoveX(1);
    CudaUtils::Memory<size_t> cudaMoveY(1);
    
    cudaPlayfield.transferFrom(state.playfieldBuffer());

    compute(reiterations, state.playfieldSideLength(), cudaPlayfield.get(), cudaMoveX.get(), cudaMoveY.get(), cudaWins.get(), cudaVisits.get());
    return OthelloResult();
}