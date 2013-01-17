#include "OthelloNodeTest.h"
#include "OthelloState.h"
#include "OthelloMove.h"
#include "OthelloUtil.h"
#include <functional>
#include <vector>
#include <stdexcept>


using namespace std;

class OthelloNodeStub : public OthelloNode
{
public:
    OthelloNodeStub(const OthelloNode& node) : OthelloNode(node) {}
    OthelloNodeStub(const OthelloState& state) : OthelloNode(state) {}
    ~OthelloNodeStub() {}

    void setGenerator(std::function<size_t()> generator)
    {
        _generator = generator;
    }

protected:
    std::function<size_t()> _generator;

    virtual std::function<size_t()> getGenerator(size_t, size_t)
    {
        return _generator;
    }
};

#define _F Field::Free
#define _W Field::White
#define _B Field::Black

TEST_F(OthelloNodeTest, SimpleTest)
{
    OthelloState state(vector<Field>{
        _F, _F, _F, _F, _F, _F, _F, _F, 
        _F, _F, _F, _F, _F, _F, _F, _F, 
        _F, _F, _F, _F, _F, _F, _F, _F, 
        _F, _F, _F, _W, _B, _F, _F, _F, 
        _F, _F, _F, _B, _W, _F, _F, _F, 
        _F, _F, _F, _F, _F, _F, _F, _F, 
        _F, _F, _F, _F, _F, _F, _F, _F, 
        _F, _F, _F, _F, _F, _F, _F, _F
    }, Player::White);
    OthelloNodeStub stub(state);
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
        _F, _F, _F, _F, _F, _F, _F, _F, 
        _F, _F, _F, _F, _F, _F, _F, _F, 
        _F, _F, _F, _F, _F, _F, _F, _F, 
        _F, _F, _F, _W, _B, _F, _F, _F, 
        _F, _F, _F, _B, _W, _F, _F, _F, 
        _F, _F, _F, _F, _F, _F, _F, _F, 
        _F, _F, _F, _F, _F, _F, _F, _F, 
        _F, _F, _F, _F, _F, _F, _F, _F
    }, Player::White);
    OthelloNodeStub stub(state);
    stub.setGenerator([](){ return 0U; });
    auto possibleMoves = state.getPossibleMoves();
    ASSERT_EQ(possibleMoves[0], stub.getRandomUntriedMove());
    state.doMove(possibleMoves[0]);
    stub.addChild(possibleMoves[0], OthelloState(state));
    ASSERT_TRUE(stub.hasChildren());

    auto child = stub.getRandomChildNode();
    ASSERT_EQ(possibleMoves[0], child.getTriggerMove());
    child.updateSuccessProbability(true);
    child.updateSuccessProbability(false);
    ASSERT_EQ(0.5, child.successRate());

    ASSERT_NE(stub.currentPlayer(), child.currentPlayer());
    ASSERT_EQ(&(stub), &(child.parent()));
}

TEST_F(OthelloNodeTest, SelectFavoriteChild)
{
    OthelloState state(vector<Field>{
        _F, _F, _F, _F, _F, _F, _F, _F, 
        _F, _F, _F, _F, _F, _F, _F, _F, 
        _F, _F, _F, _F, _F, _F, _F, _F, 
        _F, _F, _F, _W, _B, _F, _F, _F, 
        _F, _F, _F, _B, _W, _F, _F, _F, 
        _F, _F, _F, _F, _F, _F, _F, _F, 
        _F, _F, _F, _F, _F, _F, _F, _F, 
        _F, _F, _F, _F, _F, _F, _F, _F
    }, Player::White);

    OthelloNodeStub root(state);
    root.setGenerator([](){ return 0U; });
    
    auto possibleMoves = state.getPossibleMoves();
    state.doMove(possibleMoves[0]);
    root.addChild(possibleMoves[0], OthelloState(state));

    auto looser = root.getRandomChildNode();
    looser.updateSuccessProbability(true);
    looser.updateSuccessProbability(false);
    ASSERT_EQ(0.5, looser.successRate());

    root.setGenerator([](){ return 1U; });
    
    state.doMove(possibleMoves[1]);
    root.addChild(possibleMoves[1], OthelloState(state));

    auto winner = root.getRandomChildNode();
    winner.updateSuccessProbability(true);
    ASSERT_EQ(1, winner.successRate());

    ASSERT_EQ(&(winner), &(root.getFavoriteChild()));
}
