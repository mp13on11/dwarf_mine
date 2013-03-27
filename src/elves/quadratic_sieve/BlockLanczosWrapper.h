#pragma once

#include "QuadraticSieve.h"
#include <vector>

namespace BlockLanczosWrapper
{
    class BlockLanczosResult;

    BlockLanczosResult blockLanczos(
        const std::vector<Relation>& relations, 
        const FactorBase& factorBaseSize,
        const BigInt& number
    );

    std::vector<BigInt> findFactors(BlockLanczosResult& result);
}
