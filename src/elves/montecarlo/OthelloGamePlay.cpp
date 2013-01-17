#include "OthelloGamePlay.h"
#include "OthelloNode.h"

OthelloMove OthelloGamePlay::getBestMoveFor(OthelloState& state, size_t iterations)
{
	OthelloNode node(state);


	for(size_t i = 0; i < iterations; ++i)
	{
		auto currentNode(node);
		auto currentState(state);

		//SELECT
		while(!currentNode.hasUntriedMoves() && currentNode.hasChildren())
		{
			currentNode = currentNode.getRandomChildNode();
			OthelloMove move = currentNode.getTriggerMove();
			currentState.doMove(move);
		}

		//EXPAND
		if(currentNode.hasUntriedMoves())
		{
			OthelloMove move = currentNode.getRandomUntriedMove();
			currentState.doMove(move);
			currentNode = currentNode.addChild(move, currentState);
		}

		//ROLLOUT
		while(currentState.hasPossibleMoves())
		{
			OthelloMove move = currentNode.getRandomMove();
			currentState.doMove(move);
		}

		//BACKPROPAGATION
		do
		{
			currentNode.updateSuccessProbability(currentState.hasWon(currentNode.currentPlayer()));
			currentNode = currentNode.parent();
		}
		while(currentNode.hasParent());
	}

	return node.getFavoriteChild().getTriggerMove();
}