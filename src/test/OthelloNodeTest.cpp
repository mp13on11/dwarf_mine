#include "OthelloNodeTest.h"
#include "OthelloState.h"
#include "OthelloMove.h"
#include "OthelloUtil.h"
#include <functional>
#include <vector>
#include <stdexcept>


using namespace std;

TEST_F(OthelloNodeTest, SimpleTest)
{
    OthelloState state(vector<Field>{
        F, F, F, F, F, F, F, F, 
        F, F, F, F, F, F, F, F, 
        F, F, F, F, F, F, F, F, 
        F, F, F, W, B, F, F, F, 
        F, F, F, B, W, F, F, F, 
        F, F, F, F, F, F, F, F, 
        F, F, F, F, F, F, F, F, 
        F, F, F, F, F, F, F, F
    }, Player::White);
    OthelloNode stub(state);
    ASSERT_FALSE(stub.hasChildren());
    ASSERT_THROW(stub.getTriggerMove(), runtime_error);

    ASSERT_TRUE(stub.hasUntriedMoves());
    ASSERT_EQ(state.getCurrentPlayer(), stub.currentPlayer());
    ASSERT_FALSE(stub.hasParent());
    ASSERT_EQ(0, stub.successRate());
    
}

TEST_F(OthelloNodeTest, MakeMoveTest)
{
    OthelloState state(vector<Field>{
        F, F, F, F, F, F, F, F, 
        F, F, F, F, F, F, F, F, 
        F, F, F, F, F, F, F, F, 
        F, F, F, W, B, F, F, F, 
        F, F, F, B, W, F, F, F, 
        F, F, F, F, F, F, F, F, 
        F, F, F, F, F, F, F, F, 
        F, F, F, F, F, F, F, F
    }, Player::White);
    OthelloNode stub(state);
    auto generator = [](size_t){ return 0U; };

    auto possibleMoves = state.getPossibleMoves();
    ASSERT_EQ(possibleMoves[0], stub.getRandomUntriedMove(generator));
    
    state.doMove(possibleMoves[0]);
    stub.addChild(possibleMoves[0], OthelloState(state));
    ASSERT_TRUE(stub.hasChildren());

    auto child = stub.getRandomChildNode(generator);
    ASSERT_EQ(possibleMoves[0], child->getTriggerMove());
    
    child->updateSuccessProbability(true);
    child->updateSuccessProbability(false);
    ASSERT_EQ(0.5, child->successRate());

    ASSERT_NE(stub.currentPlayer(), child->currentPlayer());
    ASSERT_EQ(&(stub), &(child->parent()));
}

TEST_F(OthelloNodeTest, SelectFavoriteChild)
{
    OthelloState state(vector<Field>{
        F, F, F, F, F, F, F, F, 
        F, F, F, F, F, F, F, F, 
        F, F, F, F, F, F, F, F, 
        F, F, F, W, B, F, F, F, 
        F, F, F, B, W, F, F, F, 
        F, F, F, F, F, F, F, F, 
        F, F, F, F, F, F, F, F, 
        F, F, F, F, F, F, F, F
    }, Player::White);

    OthelloNode root(state);
   
    auto possibleMoves = state.getPossibleMoves();
    state.doMove(possibleMoves[0]);
    OthelloNode* looser = root.addChild(possibleMoves[0], OthelloState(state));

    looser->updateSuccessProbability(true);
    looser->updateSuccessProbability(false);
    ASSERT_EQ(0.5, looser->successRate());

    state.doMove(possibleMoves[1]);
    OthelloNode* winner = root.addChild(possibleMoves[1], OthelloState(state));

    winner->updateSuccessProbability(true);
    ASSERT_EQ(1, winner->successRate());

    ASSERT_EQ(winner, root.getFavoriteChild());
}
