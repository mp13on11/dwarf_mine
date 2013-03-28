#pragma once

#include "CudaUtil.cuh"
#include "CudaGameState.cuh"
#include "CudaDebug.cuh"

class CudaSimulator
{
private:
    size_t _playfieldIndex;
    size_t _playfieldX;
    size_t _playfieldY;
    CudaGameState* _state;
    curandState* _deviceState;
	size_t _randomSeed;
	float* _randomValues;
    size_t _stepCounter;

public:
    __device__ CudaSimulator(CudaGameState* state, curandState* deviceState)
        : _playfieldIndex(threadIdx.x), _playfieldX(_playfieldIndex % FIELD_DIMENSION), _playfieldY(_playfieldIndex / FIELD_DIMENSION),
            _state(state), _deviceState(deviceState), _stepCounter(0)
    {
    }

    __device__ CudaSimulator(CudaGameState* state, float* randomValues, size_t randomSeed)
        : _playfieldIndex(threadIdx.x), _playfieldX(_playfieldIndex % FIELD_DIMENSION), _playfieldY(_playfieldIndex / FIELD_DIMENSION),
            _state(state), _randomSeed(randomSeed), _randomValues(randomValues), _stepCounter(0)
    {
    }

    __device__ void expandLeaf()
    {
        size_t passCounter = 0;

        __syncthreads();
        
        while (passCounter < 2)
        {
            bool passedMove = !doStep();
            passCounter = (passedMove ? passCounter + 1 : 0);

            cassert (_stepCounter < MAXIMAL_MOVE_COUNT, "Detected rounds overflowing maximal count %d in %d\n", MAXIMAL_MOVE_COUNT, threadIdx.x); 
            ++_stepCounter;
        }
        __syncthreads();
    }

    __device__ bool doStep(float fakedRandom = -1)
    {
        cassert(state.size == FIELD_DIMENSION * FIELD_DIMENSION, "Block %d, Thread %d detected invalid field size of %li\n", blockIdx.x, threadIdx.x, state.size);
        
        calculatePossibleMoves();
        
        size_t moveCount = countPossibleMoves();
        
        if (moveCount > 0)
        {
            size_t index = getRandomMoveIndex(moveCount, fakedRandom);
            cassert(index < state.size, "Block %d, Thread %d: Round %d detected unexpected move index %d for maximal playfield size %lu\n", blockIdx.x, _stepCounter, index, state.size);

            flipEnemyCounter(index);

            cassert(!_state.isUnchanged(), "Block %d: %lu detected unchanged state\n", blockIdx.x, _stepCounter);
        }

        _state->switchPlayer();
        return moveCount > 0;
    }

    __device__ bool isMaster()
    {
        return _playfieldIndex == 0;
    }
    
    __device__ void calculatePossibleMoves()
    {
        __syncthreads();
        
        _state->possible[_playfieldIndex] = false;
    
        __syncthreads();

        if (_state->field[_playfieldIndex] == Free)
        {
            findPossibleMoves( 1,  1);
            findPossibleMoves( 1,  0);
            findPossibleMoves( 1, -1);
            findPossibleMoves( 0,  1);
            
            findPossibleMoves( 0, -1);
            findPossibleMoves(-1,  1);
            findPossibleMoves(-1,  0);
            findPossibleMoves(-1, -1);
        }

        __syncthreads();
    }

    __device__ void findPossibleMoves(int directionX, int directionY)
    {
        bool foundEnemy = false;
        Player enemyPlayer = _state->getEnemyPlayer();
        int neighbourX = _playfieldX + directionX;
        int neighbourY = _playfieldY + directionY;
        
        int neighbourIndex = neighbourY * FIELD_DIMENSION + neighbourX;
        while (_state->inBounds(neighbourX, neighbourY) && _state->field[neighbourIndex] == enemyPlayer)
        {
            foundEnemy = true;
            
            neighbourX += directionX;
            neighbourY += directionY;
            neighbourIndex = neighbourY * FIELD_DIMENSION + neighbourX;
        }

        if (_state->inBounds(neighbourX, neighbourY) && _state->field[neighbourIndex] == _state->currentPlayer)
        {
            if (foundEnemy)
                _state->possible[_playfieldIndex] |= foundEnemy;      
        }
    }

    __device__ size_t countPossibleMoves()
    {
        return _state->numberOfMarkedFields();
    }

    // this function may deliver different results for the threads, so it should be only called once per block
    __device__ size_t getRandomMoveIndex(size_t moveCount, float fakedRandom = -1)
    {
        __shared__ size_t randomMoveCounter;
        if (_playfieldIndex == 0)
        {
            if (moveCount > 1)
            {
                //randomMoveCounter = randomNumber(_deviceState, moveCount, fakedRandom);    
				randomMoveCounter = randomNumber(_randomValues, &_randomSeed, moveCount, fakedRandom); 
            }
            else
            {
                randomMoveCounter = 0;
            }
        }
        __syncthreads();
        cassert(randomMoveCounter < moveCount, "Detected invalid moveCounter %lu - maximum %lu\n", randomMoveCounter, moveCount);

        size_t possibleMoveCounter = 0;
        for (size_t i = 0; i < _state->size; ++i)
        {
            if (_state->possible[i])
            {
                if (possibleMoveCounter == randomMoveCounter)
                {
                    return i;
                }
                possibleMoveCounter++;
            }
        }
        cassert(false, "Could not find array index for move #%d - found only %lu possible moves in field of size %lu\n", randomMoveCounter, possibleMoveCounter, _state->size);
        return 96;
    }

    __device__ bool canFlipInDirection(size_t moveIndex, int directionX, int directionY)
    {
        Player enemyPlayer = _state->getEnemyPlayer();

        for (size_t currentIndex = _playfieldIndex; _state->inBounds(currentIndex); currentIndex += directionY * _state->sideLength + directionX)
        {
            if(_state->field[currentIndex] != enemyPlayer)
            {
                return (_state->field[currentIndex] == _state->currentPlayer && currentIndex != _playfieldIndex);
            }
        }
        return false;
    }

    __device__ void flipInDirection(size_t moveIndex, int directionX, int directionY)
    {
        Player enemyPlayer = _state->getEnemyPlayer();

        for (size_t currentIndex = _playfieldIndex; _state->inBounds(currentIndex); currentIndex += directionY * _state->sideLength + directionX)
        {
            if(_state->oldField[currentIndex] == enemyPlayer)
            {
                _state->field[currentIndex] = _state->currentPlayer;
            }
            else
            {
                break;
            }
        }
    }

    __device__ void flipEnemyCounter(size_t moveIndex)
    {
        __syncthreads();

        _state->oldField[_playfieldIndex] = _state->field[_playfieldIndex];

        __syncthreads();

        int directionX = _playfieldX - moveIndex % _state->sideLength;
        int directionY = _playfieldY - moveIndex / _state->sideLength;

        bool flip = false;
        if (abs(directionX) <= 1 && abs(directionY) <= 1 && moveIndex != _playfieldIndex)
        {
            flip = canFlipInDirection(moveIndex, directionX, directionY);
        }

        __syncthreads();

        if (flip)
        {
            flipInDirection(moveIndex, directionX, directionY);
        }

        __syncthreads();

        if (_playfieldIndex == moveIndex && threadIdx.x == moveIndex)
        {
            _state->field[_playfieldIndex] = _state->currentPlayer;
        }
    }
};
