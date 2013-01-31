#pragma once

#include "Move.cuh"

class State
{
private:
	Field* _playfield;
	size_t _dimension;
	Player _currentPlayer;
	size_t _visits;
	size_t _wins;
	Move _triggerMove;
public:
	__device__ State()
	{

	}

	__device__ State(Field* playfield, size_t dimension, Player startPlayer)
	{
		_dimension = dimension;
		_currentPlayer = startPlayer;
		_playfield = new Field[dimension * dimension];
		memcpy(_playfield, playfield, dimension * dimension * sizeof(Field));
	}

	__device__ State(const State& state, const Move& move)
	{
		_dimension = state._dimension;
		_playfield = new Field[_dimension * _dimension];
		memcpy(_playfield, state._playfield, _dimension * _dimension * sizeof(Field));
		_playfield[move.y * _dimension + move.x] = state.currentPlayer();
		_currentPlayer = state.enemyPlayer();
		_triggerMove = move;
	}

	__device__ size_t getTriggerMoveX()
	{
		return _triggerMove.x;
	}

	__device__ size_t getTriggerMoveY()
	{
		return _triggerMove.y;
	}

	__device__ ~State()
	{
		if (_playfield != NULL)
		{
			delete[] _playfield;
		}
	}

	__device__ Player currentPlayer() const
	{
		return _currentPlayer;
	}

	__device__ Player enemyPlayer() const
	{
		if (_currentPlayer == White)
			return Black;
		return White;
	}

	__device__ MoveVector getPossibleMoves()
	{
		// TODO calculate moves
		return MoveVector(7);
	}

	__device__ void doMove(const Move& move)
	{
		_playfield[move.y * _dimension + move.x] = currentPlayer();
		_currentPlayer = enemyPlayer();
	}

	__device__ void updateFor(Player player)
	{
		_visits++;
		if (_currentPlayer == player)
		{
			_wins++;
		}
	}

	__device__ bool hasPossibleMoves()
	{
		// TODO look for a move
		return false;
	}

	__device__ bool isGameActive()
	{
		return hasPossibleMoves();
	}

	__device__ Player getWinner()
	{
		//TODO calculate winner
		return _currentPlayer;
	}

	__device__ size_t getVisits()
	{
		return _visits;
	}

	__device__ size_t getWinsFor(Player player)
	{
		if (player == _currentPlayer)
		{
			return _wins;
		}
		return _visits - _wins;
	}
};

