#pragma once

#include "MpiUtils.h"
#include <map>

typedef int Rating; // TODO
typedef std::pair<NodeId, Rating> NodeRating;
typedef std::map<NodeId, Rating> BenchmarkResult;
