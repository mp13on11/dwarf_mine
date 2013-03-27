#pragma once

#include "QuadraticSieve.h"
#include <vector>

namespace BlockLanczosWrapper
{
    class BlockLanczosResult;

    std::vector<BigInt> performBlockLanczosAndFindFactors(
        const std::vector<Relation>& relations,
        const FactorBase& factorBaseSize,
        const BigInt& number
    );

    BlockLanczosResult blockLanczos(
        const std::vector<Relation>& relations, 
        const FactorBase& factorBaseSize,
        const BigInt& number
    );

    std::vector<BigInt> findFactors(BlockLanczosResult& result, const FactorBase& factorBase);
}
