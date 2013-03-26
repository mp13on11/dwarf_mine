#pragma once

#include "Relation.h"
#include <vector>

namespace BlockLanczosWrapper
{
    void performBlockLanczos(
        const std::vector<Relation>& relations, 
        size_t factorBaseSize
    );
}
