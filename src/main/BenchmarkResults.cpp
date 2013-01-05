

#include "BenchmarkResults.h"
#include <iostream>

std::ostream& operator<<(std::ostream& o, BenchmarkResult& results)
{
    for (const auto& result : results)
    {
        o << result.first << " " << result.second << "\n";
    }
    return o;
}

std::istream& operator>>(std::istream& i, BenchmarkResult& results)
{
    while (i.good())
    {
        NodeId nodeId;
        Rating rating;
        i >> nodeId;
        i >> rating;
        results[nodeId] = rating;
    }
    return i;
}