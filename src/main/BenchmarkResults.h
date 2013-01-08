#pragma once

#include "MpiHelper.h"

#include <map>
#include <iosfwd>

typedef int Rating; // TODO
typedef std::pair<NodeId, Rating> NodeRating;
typedef std::map<NodeId, Rating> BenchmarkResult;

std::ostream& operator<<(std::ostream& o, BenchmarkResult& results);
std::istream& operator>>(std::istream& k, BenchmarkResult& results);
