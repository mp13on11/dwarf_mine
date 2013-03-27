#include "BlockLanczosWrapper.h"
#include <unordered_map>

using namespace std;

namespace BlockLanczosWrapper 
{

    BlockLanczosResult performBlockLanczos(
        const std::vector<Relation>& relations,
        const FactorBase& factorBase,
        const BigInt& number
    )
    {
        BlockLanczosResult lanczos(relations, factorBase, number);
        size_t factorBaseSize = factorBase.size();
        uint32_t numCycles = relations.size();

        qs_solve_linear_system(
            lanczos.factorizationObject.get(),
            factorBaseSize, 
            &lanczos.bitfield,
            relationList.get(),
            lanczos.cycleList,
            &numCycles
        );
    }

}
