#include "MonteCarloElf.h"
#include <vector>

using namespace std;

void MonteCarloElf::run(istream& input, ostream& output)
{
    vector<Field> playfield;
    OthelloHelper::readPlayfieldFromStream(input, playfield);
    OthelloState state(playfield, Player::White);
    OthelloResult result = getBestMoveFor(state, 100U);
    OthelloHelper::writeResultToStream(output, result);
}
