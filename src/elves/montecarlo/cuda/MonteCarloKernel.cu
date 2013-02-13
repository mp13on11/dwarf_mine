#include "MonteCarloTreeSearch.h"
#include <curand.h>
#include <curand_kernel.h>
#include "OthelloField.h"
#include <stdio.h>

const int FIELD_DIMENSION = 8;

__global__ void setupStateForRandom(curandState* state, unsigned long seed)
{
	int id = 0; // threadIdx.x;
	curand_init(seed, id, 0, &state[id]);
}

__device__ size_t randomNumber(curandState* deviceStates, size_t maximum)
{
	curandState deviceState = deviceStates[blockIdx.x];
	size_t value = curand_uniform(&deviceState) * maximum;
	deviceStates[0] = deviceState;
    return value;
}   

typedef struct _CudaMove
{
    int x;
    int y;
} CudaMove;

typedef struct _CudaGameState
{
    Field* field;
    bool* possible;
    size_t size;
    size_t sideLength;
    Player currentPlayer;

    __device__ inline bool inBounds(int x, int y)
    {
        return (x >= 0 && x < sideLength && y >= 0 && y < sideLength);
    }

    __device__ inline bool inBounds(int i)
    {
        return (i >= 0 && i < size);
    }

    __device__ inline Player getEnemyPlayer(Player currentPlayer)
    {
        return (currentPlayer == Black ? White : Black);
    }

    __device__ inline Player getEnemyPlayer()
    {
        return getEnemyPlayer(currentPlayer);
    }

    __device__ bool isWinner(Player requestedPlayer)
    {
        Player enemyPlayer = getEnemyPlayer(requestedPlayer);
        size_t requestedCounters = 0;
        size_t enemyCounters = 0;
        for (size_t i = 0; i < size; ++i)
        {
            if (field[i] == enemyPlayer) ++enemyCounters;
            if (field[i] == requestedPlayer) ++requestedCounters;
        }
        return requestedPlayer >= enemyPlayer;
    }
} CudaGameState;


class CudaSimulator
{
private:
    size_t _playfieldIndex;
    size_t _playfieldX;
    size_t _playfieldY;
    CudaGameState* _state;
    curandState* _deviceState;

public:
    __device__ CudaSimulator(CudaGameState* state, curandState* deviceState)
        : _playfieldIndex(threadIdx.x), _playfieldX(_playfieldIndex % FIELD_DIMENSION), _playfieldY(_playfieldIndex / FIELD_DIMENSION),
            _state(state), _deviceState(deviceState)
    {
    }

    __device__ bool isMaster()
    {
        return _playfieldIndex == 0;
    }
    
    __device__ void calculatePossibleMoves()
    {
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
        bool look = true;
        bool foundEnemy = false;
        Player enemyPlayer = _state->getEnemyPlayer();
        int neighbourX = _playfieldX + directionX;
        int neighbourY = _playfieldY + directionY;
        while (look)
        {
            int neighbourIndex = neighbourY * FIELD_DIMENSION + neighbourX;
            if (_state->inBounds(neighbourX, neighbourY))
            {
                if (_state->field[neighbourIndex] == Free)
                {
                    _state->possible[_playfieldIndex] |= false;
                    look = false;
                }
                else if(_state->field[neighbourIndex] == enemyPlayer)
                {
                    foundEnemy = true;
                }
                else if (_state->field[neighbourIndex] == _state->currentPlayer)
                {
                    _state->possible[_playfieldIndex] |= foundEnemy;
                    look = false;
                }
            }
            else
            {
                _state->possible[_playfieldIndex] |= false;
                look = false;
            }
            neighbourX += directionX;
            neighbourY += directionY;
        }
    }

    __device__ size_t countPossibleMoves()
    {
        // __syncthreads();
     //    __shared__ size_t moves[FIELD_DIMENSION * FIELD_DIMENSION];
     //    moves[playfieldIndex] = possibleMoves[playfieldIndex] ? 1 : 0;
        // return sum(moves, playfieldIndex, FIELD_DIMENSION * FIELD_DIMENSION);
        
        __syncthreads();
        size_t sum = 0;
        for (int i = 0; i < _state->size; i++)
        {
            if (_state->possible[i])
            {
                sum++;
                __syncthreads();
            }
        }
        return sum;
    }

    __device__ size_t getRandomMoveIndex(size_t moveCount, float fakedRandom = -1)
    {
        size_t randomMoveIndex = 0;
        if (moveCount > 1)
        {
            if (fakedRandom >= 0)
            {
                randomMoveIndex = fakedRandom * moveCount;
            }
            else
            {
                randomMoveIndex = randomNumber(_deviceState, moveCount);    
            }
        }
        size_t possibleMoveIndex = 0;
        for (size_t i = 0; i < _state->size; ++i)
        {
            if (_state->possible[i])
            {
                if (possibleMoveIndex == randomMoveIndex)
                {
                    return i;
                }
                possibleMoveIndex++;;
            }
        }
        return 0;
    }

    __device__ void flipDirection(size_t moveIndex, int directionX, int directionY)
    {
        int currentIndex = _playfieldIndex;
        Player enemyPlayer = _state->getEnemyPlayer();
        bool flip = false;

        for (currentIndex = _playfieldIndex; _state->inBounds(currentIndex); currentIndex += directionY * _state->sideLength + directionX)
        {
            if(_state->field[currentIndex] != enemyPlayer)
            {
                flip = (_state->field[currentIndex] == _state->currentPlayer && currentIndex != _playfieldIndex);
                break;
            }
        }
        __syncthreads();
        if (flip)
        {
            for (; currentIndex - moveIndex != 0 ; currentIndex -= directionY * _state->sideLength + directionX)
            {
                _state->field[currentIndex] = _state->currentPlayer;
            }
        }
    }

    __device__ void flipEnemyCounter(size_t moveIndex)
    {
        int directionX = _playfieldX - moveIndex % _state->sideLength;
        int directionY = _playfieldY - moveIndex / _state->sideLength;

        if (abs(directionX) <= 1 && abs(directionY) <= 1 && moveIndex != _playfieldIndex)
        {
            flipDirection(moveIndex, directionX, directionY);
        }
    }
};

__device__ bool doStep(CudaGameState& state, CudaSimulator& simulator, float fakedRandom = -1)
{
    __syncthreads();

    simulator.calculatePossibleMoves();
    size_t moveCount = simulator.countPossibleMoves();
    if (threadIdx.x == 0)
        printf("Block %d: Moves %lu \n", blockIdx.x, moveCount);
    if (moveCount > 0)
    {
        __shared__ size_t index;
        if (threadIdx.x == 0)
            index = simulator.getRandomMoveIndex(moveCount, fakedRandom);
        
        __syncthreads();

        simulator.flipEnemyCounter(index);

        __syncthreads();
        state.field[index] = state.currentPlayer;
    }
    state.currentPlayer = state.getEnemyPlayer();
   
    return moveCount > 0;
}

__device__ void simulateGameLeaf(curandState* deviceState, CudaSimulator& simulator, CudaGameState& state, size_t* wins, size_t* visits)
{
    Player startingPlayer = state.currentPlayer;
    size_t passCounter = 0;
    size_t limit = 0;
    while (limit < 128)
    {
        bool result = !doStep(state, simulator);
        if (result)
        {
            passCounter++;
            //__syncthreads();
            // if (threadIdx.x == 0)
            //     printf("Block %d: Raise counter to %lu on %lu\n", blockIdx.x, passCounter, limit);
            if (passCounter > 1)
                break;
        }
        else
        {
            passCounter = 0;
            // __syncthreads();
            // if (threadIdx.x == 0)
            //     printf("Block %d: Reset counter to %lu on %lu\n", blockIdx.x, passCounter, limit);
        }
        limit++;
    }
    if (threadIdx.x == 0)
    if (passCounter < 2)
        printf("Block %d unexpected exited game\n", blockIdx.x);
    else
        printf("Block %d exited game\n", blockIdx.x);
    __syncthreads();

    ++(*visits);
    if (state.isWinner(startingPlayer))
    {
        ++(*wins);
    }
    
}

__global__ void simulateGameLeaf(curandState* deviceState, Field* playfield, Player currentPlayer, size_t* wins, size_t* visits)
{
    int playfieldIndex = threadIdx.x;

    __shared__ Field sharedPlayfield[FIELD_DIMENSION * FIELD_DIMENSION];
    __shared__ bool possibleMoves[FIELD_DIMENSION*FIELD_DIMENSION];
    sharedPlayfield[playfieldIndex] = playfield[playfieldIndex];

    CudaGameState state =  { 
        sharedPlayfield, 
        possibleMoves, 
        FIELD_DIMENSION * FIELD_DIMENSION, 
        FIELD_DIMENSION, 
        currentPlayer 
    };
    CudaSimulator simulator(&state, deviceState);
    simulateGameLeaf(deviceState, simulator, state, wins, visits);
}

__global__ void simulateGame(size_t reiterations, curandState* deviceStates, size_t numberOfPlayfields, Field* playfields, Player currentPlayer, OthelloResult* results)
{
    __shared__ size_t node;
    int threadGroup = blockIdx.x;
    int playfieldIndex = threadIdx.x;
    if (threadIdx.x == 0) 
    {
        node = randomNumber(deviceStates, numberOfPlayfields);
    }

    __shared__ Field sharedPlayfield[FIELD_DIMENSION * FIELD_DIMENSION];
    __shared__ bool possibleMoves[FIELD_DIMENSION*FIELD_DIMENSION];
    
    size_t playfieldOffset = FIELD_DIMENSION * FIELD_DIMENSION * threadGroup;
    sharedPlayfield[playfieldIndex] = playfields[playfieldOffset + playfieldIndex];

    CudaGameState state =  { 
        sharedPlayfield, 
        possibleMoves, 
        FIELD_DIMENSION * FIELD_DIMENSION, 
        FIELD_DIMENSION, 
        currentPlayer 
    };
    CudaSimulator simulator(&state, deviceStates);
    OthelloResult& result = results[node];
    //OthelloResult result;
    size_t wins = 0;
    size_t visits = 0;
        if (threadIdx.x == 0)
        printf("Block %d started game on %lu\n", blockIdx.x, node);
    __syncthreads();
    simulateGameLeaf(deviceStates, simulator, state, &wins, &visits);
    __syncthreads();
        
    __syncthreads();
    if (threadIdx.x == 0)
    {
        results[node].wins += wins;
        results[node].visits += visits;
        //printf("TEST %d\n", result.wins);
        printf("Block %d finished game on %lu with %lu wins in %lu visits \n", blockIdx.x, node, results[node].wins, results[node].visits);
    }
}

__global__ void testDoStep(curandState* deviceState, Field* playfield, Player currentPlayer, float fakedRandom)
{
    int playfieldIndex = threadIdx.x;
    __shared__ Field sharedPlayfield[FIELD_DIMENSION * FIELD_DIMENSION];
    __shared__ bool possibleMoves[FIELD_DIMENSION*FIELD_DIMENSION];
    sharedPlayfield[playfieldIndex] = playfield[playfieldIndex];

    CudaGameState state =  { 
        sharedPlayfield, 
        possibleMoves, 
        FIELD_DIMENSION * FIELD_DIMENSION, 
        FIELD_DIMENSION, 
        currentPlayer 
    };
    CudaSimulator simulator(&state, deviceState);

    doStep(state, simulator, fakedRandom);

    playfield[playfieldIndex] = sharedPlayfield[playfieldIndex];
}

__global__ void testSimulateGameLeaf(curandState* deviceState, Field* playfield, Player currentPlayer, size_t* wins, size_t* visits)
{
    int playfieldIndex = threadIdx.x;

    __shared__ Field sharedPlayfield[FIELD_DIMENSION * FIELD_DIMENSION];
    __shared__ bool possibleMoves[FIELD_DIMENSION*FIELD_DIMENSION];
    sharedPlayfield[playfieldIndex] = playfield[playfieldIndex];

    CudaGameState state =  { 
        sharedPlayfield, 
        possibleMoves, 
        FIELD_DIMENSION * FIELD_DIMENSION, 
        FIELD_DIMENSION, 
        currentPlayer 
    };
    CudaSimulator simulator(&state, deviceState);
    simulateGameLeaf(deviceState, simulator, state, wins, visits);
    
	playfield[playfieldIndex] = sharedPlayfield[playfieldIndex];
}