#include "BlockLanczosWrapper.h"

extern "C"
{
#include "lanczos_msieve/qs.h"
}

#include <unordered_map>

using namespace std;

namespace BlockLanczosWrapper 
{
    
    qs_la_col_t* makeCycleList(const vector<Relation>& relations)
    {
        qs_la_col_t* list = reinterpret_cast<qs_la_col_t*>(malloc(sizeof(qs_la_col_t) * relations.size()));

        for (size_t i=0; i<relations.size(); ++i)
        {
            list[i].cycle.num_relations = 1;
            list[i].cycle.list = reinterpret_cast<uint32*>(malloc(sizeof(uint32)));
            list[i].cycle.list[0] = i;
        }

        return list;
    }

    unique_ptr<siqs_r[]> makeRelationList(const vector<Relation>& relations, const FactorBase& factorBase, const BigInt& sieveStart)
    {
        unordered_map<smallPrime_t, size_t> factorMapping;

        for (size_t i=0; i<factorBase.size(); ++i)
        {
            factorMapping[factorBase[i]] = i;
        }

        unique_ptr<siqs_r[]> list(new siqs_r[relations.size()]);
        size_t index = 0;
        for (const auto& relation : relations)
        {
            list[index].poly_idx = 0;
            list[index].parity = 0;
            BigInt sieveOffset = relation.a - sieveStart;
            list[index].sieve_offset = sieveOffset.get_ui();
            list[index].large_prime[0] = 1;
            list[index].large_prime[1] = 1;
            list[index].num_factors = relation.oddPrimePowers.indices.size();
            list[index].fb_offsets = new uint32_t[list[index].num_factors];

            for (size_t i=0; i<relation.oddPrimePowers.indices.size(); ++i)
            {
                list[index].fb_offsets[i] = factorMapping[relation.oddPrimePowers.indices[i]];
            }
            ++index;
        }
        return list;
    }

#define DEFAULT_L1_CACHE_SIZE (32 * 1024)
#define DEFAULT_L2_CACHE_SIZE (512 * 1024)

    unique_ptr<fact_obj_t> makeFactorizationObject()
    {
        unique_ptr<fact_obj_t> factorizationObject(new fact_obj_t);
        factorizationObject->num_threads = 1;
        factorizationObject->flags = 0;
        factorizationObject->logfile = nullptr;
        factorizationObject->cache_size1 = DEFAULT_L1_CACHE_SIZE;
        factorizationObject->cache_size2 = DEFAULT_L2_CACHE_SIZE;
        //factorizationObject->bits =
        //yafu_get_cache_sizes(&factorizationObject->cache_size1, &factorizationObject->cache_size2);

        FILE *rand_device = fopen("/dev/urandom", "r");

        if (rand_device != nullptr)
        {
            fread(&factorizationObject->seed1, sizeof(uint32), (size_t)1, rand_device);
            fread(&factorizationObject->seed2, sizeof(uint32), (size_t)1, rand_device);
            fclose(rand_device);
        }
        return factorizationObject;
    }

    void performBlockLanczos(
        const std::vector<Relation>& relations,
        const FactorBase& factorBase,
        const BigInt& sieveStart
    )
    {
        size_t factorBaseSize = factorBase.size();
        uint32_t numCycles = relations.size();
        uint64_t* bitfield;
        qs_la_col_t* cycleList = makeCycleList(relations);
        unique_ptr<siqs_r[]> relationList = makeRelationList(relations, factorBase, sieveStart);
        unique_ptr<fact_obj_t> factorizationObject(makeFactorizationObject());

        qs_solve_linear_system(
            factorizationObject.get(),
            factorBaseSize, 
            &bitfield,
            relationList.get(),
            cycleList,
            &numCycles
        );
        cout << "NumCycles: " << numCycles << endl;
        cout << hex << "Bitfield" << bitfield[0] << endl;
    }

}
