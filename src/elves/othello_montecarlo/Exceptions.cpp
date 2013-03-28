#include "Exceptions.h"

#include <sstream>
#include "OthelloUtil.h"

using namespace std;

InvalidFieldSizeException::InvalidFieldSizeException(size_t sideLength) :
    logic_error(constructMessage(sideLength))
{
}

string InvalidFieldSizeException::constructMessage(size_t sideLength)
{
    stringstream stream;
    stream << "Othello field size must be a multiple of 2 - given: "<<sideLength;
    return stream.str();
}


InvalidMoveException::InvalidMoveException(size_t sideLength, const Move& move) :
    runtime_error(constructMessage(sideLength, move))
{
}

string InvalidMoveException::constructMessage(size_t sideLength, const Move& move)
{
    stringstream stream;
    stream << "Move"<<move<<" outside of playfield ["<<sideLength<<"x"<<sideLength<<"]";
    return stream.str();
}


OccupiedFieldException::OccupiedFieldException( const Move& move) :
    runtime_error(constructMessage(move))
{
}

string OccupiedFieldException::constructMessage(const Move& move)
{
    stringstream stream;
    stream << "Move"<<move<<" on occupied field";
    return stream.str();
}