#include "OthelloUtil.h"
#include <OthelloResult.h>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>

namespace OthelloHelper
{
    size_t generateUniqueSeed(size_t nodeId, size_t threadId, size_t commonSeed)
    {
        std::stringstream buffer;
        buffer << commonSeed << " ~ "<< nodeId << " ~ " << threadId;
        std::hash<std::string> generateHash;
        return generateHash(buffer.str());
    }

    void writeResultToStream(std::ostream& stream, const OthelloResult& result)
    {
        stream << result.x << result.y << result.visits << result.wins << result.iterations;
    }

    void readResultFromStream(std::istream& stream, OthelloResult& result)
    {
        stream >> result.x;
        stream >> result.y;
        stream >> result.visits;
        stream >> result.wins;
        stream >> result.iterations;
    }

    void writePlayfieldToStream(std::ostream& stream, const std::vector<Field>& playfield)
    {
        for (const auto& field : playfield)
        {
            stream << field;
        }
    }

    void readPlayfieldFromStream(std::istream& stream, std::vector<Field>& playfield)
    {
        if (stream.good())
        {
            do 
            {
                Field field;
                stream >> field;
                playfield.push_back(field);
            }
            while (stream.good());
        }
        else
        {
            throw std::runtime_error("Unexpected empty stream");
        }
    }
}

std::ostream& operator<<(std::ostream& stream, const Field field)
{
    if (field == Field::Black)
    {
        stream << "B";
    }
    else if (field == Field::White)
    {
        stream << "W";
    }
    else if (field == Field::Illegal)
    {
        stream << "?";
    }
    else 
    {
        stream << "F";
    }
    return stream;
}

std::istream& operator>>(std::istream& stream, Field& field)
{
    char sign;
    stream >> sign;
    switch (sign)
    {
        case 'B': field = Field::Black;
            break;
        case 'W': field = Field::White;
            break;
        case 'F': field = Field::Free;
            break;
    }
    return stream;
}