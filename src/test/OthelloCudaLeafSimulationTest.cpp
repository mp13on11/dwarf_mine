#include "OthelloCudaLeafSimulationTest.h"
#include "OthelloState.h"
#include "OthelloMove.h"
#include "OthelloUtil.h"
#include <cuda-utils/Memory.h>
#include <functional>
#include <vector>
#include <utility>
#include <stdexcept>


using namespace std;

#define ASSERT_EQ_VECTOR(ACTUAL, EXPECTED) \
    ASSERT_EQ(ACTUAL.size(), EXPECTED.size()); \
    for (size_t i = 0; i < ACTUAL.size(); ++i) \
        ASSERT_EQ(ACTUAL[i], EXPECTED[i])<<"\t  at index: "<<i;

void testSingleStep(Playfield& playfield, vector<pair<size_t, Field>> expectedChanges, Player currentPlayer, float randomFake)
{
    CudaUtils::Memory<Field> cudaPlayfield(playfield.size());
    cudaPlayfield.transferFrom(playfield.data());
    
    testBySimulateSingeStep(cudaPlayfield.get(), currentPlayer, randomFake);

    Playfield outputPlayfield(playfield.size());
    cudaPlayfield.transferTo(outputPlayfield.data());

    Playfield expectedPlayfield;
    expectedPlayfield.assign(playfield.begin(), playfield.end());
    for (const auto change : expectedChanges)
    {
        expectedPlayfield[change.first] = change.second;
    }
    // OthelloState temp(outputPlayfield, Player::White);
    // cout << "Actual: \n"<<temp << endl;

    // OthelloState temp2(expectedPlayfield, Player::White);
    // cout << "Expected: \n"<<temp2 << endl;

    ASSERT_EQ_VECTOR(outputPlayfield, expectedPlayfield);

}

TEST_F(OthelloCudaLeafSimulationTest, SingleMoveSingleFlipTest)
{
    Playfield playfield {
        F, F, F, F, F, F, F, F, 
        F, F, F, F, F, F, F, F, 
        F, F, F, F, F, F, F, F, 
        F, F, F, W, B, F, F, F, 
        F, F, F, B, W, F, F, F, 
        F, F, F, F, F, F, F, F, 
        F, F, F, F, F, F, F, F, 
        F, F, F, F, F, F, F, F
    };

    testSingleStep(playfield, {{20, W}, {28, W}}, W, 0 * 1.0 / 4);
    testSingleStep(playfield, {{28, W}, {29, W}}, W, 1 * 1.0 / 4);
    testSingleStep(playfield, {{34, W}, {35, W}}, W, 2 * 1.0 / 4);
    testSingleStep(playfield, {{35, W}, {43, W}}, W, 3 * 1.0 / 4);

    testSingleStep(playfield, {{26, B}, {27, B}}, B, 0.25);   
}

TEST_F(OthelloCudaLeafSimulationTest, SingleMoveMultipleFlipTest)
{
    Playfield playfield {
        F, F, F, F, F, F, F, F, 
        F, F, F, F, F, F, F, F, 
        F, W, W, F, B, F, F, F, 
        B, W, B, W, W, W, W, F, 
        W, B, B, B, B, B, F, F, 
        W, W, W, W, W, W, F, F, 
        F, F, B, W, W, F, F, F, 
        F, F, B, F, F, F, F, F
    };

    testSingleStep(playfield, {{48, B}, {32, B}, {40, B}, {41, B}}, B,  10 * 1.0 / 17);   
    testSingleStep(playfield, {{53, B}, {51, B}, {52, B}, {44, B}, {45, B}}, B, 12 * 1.0 / 17);   
}