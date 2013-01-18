#include "OthelloGamePlay.h"
#include "OthelloNode.h"
#include "OthelloUtil.h"
#include <functional>
#include <random>
#include <iostream>

using namespace std;


OthelloResult OthelloGamePlay::getBestMoveFor(OthelloState& state, size_t iterations)
{
	std::mt19937 engine(time(nullptr));
	auto generator = [&engine](size_t limit){ 
		uniform_int_distribution<size_t> distribution(0, limit - 1);
		return distribution(engine);
	};

	OthelloNode node(state);
	for(size_t i = 0; i < iterations; ++i)
	{
		cout << "."<< flush;
		OthelloNode* currentNode = &node;
		OthelloState currentState = state;

		//SELECT
		while(!currentNode->hasUntriedMoves() && currentNode->hasChildren())
		{
			//cout << "SELECT" <<endl;
			currentNode = &(currentNode->getRandomChildNode(generator));
			OthelloMove move = currentNode->getTriggerMove();
			currentState.doMove(move);
		}

		//EXPAND
		if(currentNode->hasUntriedMoves())
		{
			// cout << "EXPAND" <<endl;
			OthelloMove move = currentNode->getRandomUntriedMove(generator);
			currentNode->removeFromUntriedMoves(move);
			currentState.doMove(move);
			currentNode = &(currentNode->addChild(move, currentState));
		}

		//ROLLOUT
		while(currentState.hasPossibleMoves())
		{
			OthelloMove move = currentState.getRandomMove(generator);
			currentState.doMove(move);
		}

		//BACKPROPAGATION
		do
		{
			currentNode->updateSuccessProbability(currentState.hasWon(currentNode->currentPlayer()));
			currentNode = &(currentNode->parent());
		}
		while(currentNode->hasParent());
	}

	auto result = node.getFavoriteChild().collectedResult();
	//auto children = node.getChildren();
	//for (auto& c : children)
	//{
	//	auto r = c.collectedResult();
	//	cout << r<<endl;
	//}
	//exit(1);
	return result;
}