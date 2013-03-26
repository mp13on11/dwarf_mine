#pragma once

#include "QuadraticSieve.h"
#include <vector>

namespace BlockLanczosWrapper
{
    void performBlockLanczos(
        const std::vector<Relation>& relations, 
        const FactorBase& factorBaseSize,
        const BigInt& sieveStart
    );
}
