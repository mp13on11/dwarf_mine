#pragma once

#include "OthelloMove.h"
#include <vector>
#include <memory>
#include <set>


enum class Field { Free, Black, White };

typedef Field Player;

class OthelloState
{
public:
	OthelloState(const OthelloState& state);
	OthelloState(size_t sideLength = 8);

	void doMove(const OthelloMove& move);
	std::vector<OthelloMove> getPossibleMoves();
	Player getCurrentPlayer() const;
	Player getCurrentEnemy() const;
	bool hasPossibleMoves();
	bool hasWon(Player player);

private:
	// row - column
	std::vector<Field> _playfield;
	int _sideLength;
	Player _player;

	bool onBoard(const OthelloMove& move) const;
	std::vector<OthelloMove> getAllSandwichCounters(const OthelloMove& move);
	std::vector<OthelloMove> getAdjacentEnemyDirections(const OthelloMove& move);
	std::vector<OthelloMove> getSandwichedCounters(const OthelloMove& move, const OthelloMove& direction);
	bool existsSandwichedCounters(const OthelloMove& move);
	Field& playfield(const OthelloMove& move);
};

inline Field& OthelloState::playfield(const OthelloMove& move)
{
	return _playfield[move.y * _sideLength + move.x];
}
