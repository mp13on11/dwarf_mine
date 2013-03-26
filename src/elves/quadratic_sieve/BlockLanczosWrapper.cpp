#include "BlockLanczosWrapper.h"
#include "lanczos_msieve/qs.h"

using namespace std;

namespace BlockLanczosWrapper 
{
    
    void performBlockLanczos(
        const std::vector<Relation>& relations, 
        size_t factorBaseSize
    )
    {
        uint32_t numCycles = 0; 
        uint64_t* bitfield;
        qs_la_col_t* cycleList = nullptr;

        unique_ptr<siqs_r[]> relationList = adaptRelations(relations);

        qs_solve_linear_system(
            factorizationObject, 
            factorBaseSize, 
            &bitfield,
            relationList,
            cycleList,
            &numCycles
        );
    }
    /*
     *
void qs_solve_linear_system(fact_obj_t *obj, uint32 fb_size, 
		    uint64 **bitfield, siqs_r *relation_list, 
            qs_la_col_t *cycle_list, uint32 *num_cycles) {
            */
}
