#pragma once

#include "QuadraticSieve.h"
#include <vector>

extern "C"
{
#include "lanczos_msieve/qs.h"
}

namespace BlockLanczosWrapper
{

    class BlockLanczosResult
    {
    public:
        BlockLanczosResult(
            const std::vector<Relation>& relations,
            const FactorBase& factorBase,
            const BigInt& number
        );
        ~BlockLanczosResult();

    private:
        uint64* bitfield;
        qs_la_col_t* cycleList;
        std::unique_ptr<siqs_r[]> relationList;
        std::unique_ptr<fact_obj_t> factorizationObject;
    };

}
