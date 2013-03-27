#include "BlockLanczosWrapper.h"
#include <unordered_map>

extern "C"
{
#include "lanczos_msieve/gmp_xface.h"
#include "lanczos_msieve/qs.h"

uint8 choose_multiplier_siqs(uint32 B, mpz_t n);
}

uint32 primeList[] = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987
};

template<typename ElemType>
std::ostream& operator<<(std::ostream& stream, const std::vector<ElemType>& list)
{
    stream << "[";
    bool first = true;
    for (const auto& element : list)
    {
        if (!first)
            stream << ", ";
        stream << element;
        first = false;
    }
    stream << "]";

    return stream;
}
using namespace std;

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
        BlockLanczosResult(BlockLanczosResult&& other);
        ~BlockLanczosResult();


        BigInt number;
        uint32 numCycles;
        uint64* bitfield;
        qs_la_col_t* cycleList;
        std::unique_ptr<siqs_r[]> relationList;
        std::unique_ptr<fact_obj_t> factorizationObject;
    };

    BlockLanczosResult::BlockLanczosResult(BlockLanczosResult&& other) :
        number(other.number),
        numCycles(other.numCycles),
        bitfield(other.bitfield),
        cycleList(other.cycleList),
        relationList(other.relationList.release()),
        factorizationObject(other.factorizationObject.release())
    {
        other.bitfield = nullptr;
        other.cycleList = nullptr;
    }

    BlockLanczosResult::~BlockLanczosResult()
    {
        if (bitfield != nullptr)
            free(bitfield);

        if (cycleList != nullptr)
        {
            // TODO
        }
    }

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

    unique_ptr<siqs_r[]> makeRelationList(const vector<Relation>& relations, const FactorBase& factorBase, const BigInt& number)
    {
        BigInt sieveStart = sqrt(number) + 1;
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

    unique_ptr<fact_obj_t> makeFactorizationObject(const BigInt& number)
    {
        unique_ptr<fact_obj_t> factorizationObject(new fact_obj_t);
        factorizationObject->num_threads = 1;
        factorizationObject->flags = 0;
        factorizationObject->logfile = nullptr;
        factorizationObject->cache_size1 = DEFAULT_L1_CACHE_SIZE;
        factorizationObject->cache_size2 = DEFAULT_L2_CACHE_SIZE;
        factorizationObject->bits = mpz_sizeinbase(number.get_mpz_t(), 2);
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

    BlockLanczosResult::BlockLanczosResult(
        const std::vector<Relation>& relations,
        const FactorBase& factorBase,
        const BigInt& number

    ) :
        number(number),
        numCycles(relations.size()),
        bitfield(nullptr),
        cycleList(makeCycleList(relations)),
        relationList(makeRelationList(relations, factorBase, number)),
        factorizationObject(makeFactorizationObject(number))
    {

    }

    vector<BigInt> performBlockLanczosAndFindFactors(
        const std::vector<Relation>& relations,
        const FactorBase& factorBase,
        const BigInt& number
    )
    {
        auto lanczos = blockLanczos(relations, factorBase, number);
        return findFactors(lanczos, factorBase);
    }


    BlockLanczosResult blockLanczos(
        const std::vector<Relation>& relations,
        const FactorBase& factorBase,
        const BigInt& number
    )
    {
        BlockLanczosResult lanczos(relations, factorBase, number);

        qs_solve_linear_system(
            lanczos.factorizationObject.get(),
            factorBase.size(),
            &lanczos.bitfield,
            lanczos.relationList.get(),
            lanczos.cycleList,
            &lanczos.numCycles
        );

        if (lanczos.numCycles == 0 || lanczos.cycleList == nullptr)
            throw runtime_error("Block Lanczos failed");

        return lanczos;
    }


    unique_ptr<fb_element_siqs[]> adaptFactorBase(const FactorBase& factorBase)
    {
        unique_ptr<fb_element_siqs[]> adapted(new fb_element_siqs[factorBase.size()]);
        for (size_t i=0; i<factorBase.size(); ++i)
        {
            adapted[i].prime = const_cast<uint32*>(&factorBase[i]);
        }
        return adapted;
    }

    vector<BigInt> findFactors(BlockLanczosResult& lanczos, const FactorBase& factorBase)
    {
        factor_list_t factorList;
        factorList.num_factors = 0;
        unique_ptr<fb_element_siqs[]> adaptedFactorBase(adaptFactorBase(factorBase));

        BigInt aValue(1);
        mpz_t aValuePtr = { *aValue.get_mpz_t() };
        mpz_t* aValueList = &aValuePtr;

        BigInt sqrtVal(sqrt(lanczos.number));
        BigInt bValue(sqrtVal);
        if (!mpz_perfect_square_p(lanczos.number.get_mpz_t()))
            bValue += 1;

        poly_t bValuePoly;
        bValuePoly.a_idx = 0;
        mpz_init_set(bValuePoly.b, bValue.get_mpz_t());

        poly_t* bValueList = &bValuePoly;

        spSOEprimes = primeList;
        szSOEp = 300;
        uint8 multiplier = choose_multiplier_siqs(factorBase.size(), lanczos.number.get_mpz_t()); // multiplier..?

        yafu_find_factors(
            lanczos.factorizationObject.get(),
            lanczos.number.get_mpz_t(),
            adaptedFactorBase.get(),
            factorBase.size(),
            lanczos.cycleList,
            lanczos.numCycles,
            lanczos.relationList.get(),
            lanczos.bitfield,
            multiplier,
            aValueList,
            bValueList,
            &factorList
        );

        vector<BigInt> result;
        for (size_t i=0; i<factorList.num_factors; ++i)
        {
            mpz_ptr converted = nullptr;
            mp_t2gmp(&factorList.final_factors[i]->factor, converted);
            result.push_back(BigInt(converted));
            mpz_clear(converted);
        }
        cout << result << endl;

        return result;

   /*
    *
uint32 yafu_find_factors(fact_obj_t *obj, mpz_t n,
		fb_element_siqs *factor_base, uint32 fb_size,
		qs_la_col_t *vectors, uint32 vsize,
		siqs_r *relation_list,
		uint64 *null_vectors, uint32 multiplier,
		mpz_t *poly_a_list, poly_t *poly_list,
		factor_list_t *factor_list) {
    */
    }

}
