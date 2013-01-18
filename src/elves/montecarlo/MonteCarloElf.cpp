#include "MonteCarloElf.h"
#include <vector>

using namespace std;

void MonteCarloElf::run(istream& input, ostream& output)
{
    vector<Field> playfield = {
        F, F, F, F, F, F, F, F,
        F, F, F, F, F, F, F, F,
        F, F, F, F, F, F, F, F,
        F, F, F, W, B, F, F, F,
        F, F, F, B, W, F, F, F,
        F, F, F, F, F, F, F, F,
        F, F, F, F, F, F, F, F,
        F, F, F, F, F, F, F, F
    };
    
    OthelloState state(playfield, Player::White);
    OthelloResult result = calculateBestMove(state);
    result.operator<<(output);// << result;
}
