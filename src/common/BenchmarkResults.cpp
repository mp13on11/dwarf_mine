#include "BenchmarkResults.h"
#include <iostream>
#include <boost/range/algorithm/copy.hpp>
#include <boost/range/adaptor/map.hpp>

std::ostream& operator<<(std::ostream& o, const BenchmarkResult& results)
{
    for (const auto& result : results)
    {
        o << result.first << " " << result.second << "\n";
    }
    return o;
}