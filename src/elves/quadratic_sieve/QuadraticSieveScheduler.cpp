#include "QuadraticSieve.h"
#include "QuadraticSieveScheduler.h"
#include "QuadraticSieveElf.h"
#include "common/ProblemStatement.h"
#include "common/Utils.h"
#include <mpi.h>
#include <numeric>

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
using namespace std::placeholders;

BigInt receiveBigInt();
BigInt receiveBigIntFromMaster();

struct serializedBigIntDeleter
{
    void operator()(uint32_t* memory)
    {
        free(memory);
    }
};

typedef std::pair<std::unique_ptr<uint32_t, serializedBigIntDeleter>, size_t> SerializedBigInt;

inline SerializedBigInt bigIntToArray(const BigInt& number)
{
    size_t arraySize;
    std::unique_ptr<uint32_t, serializedBigIntDeleter> pointer(reinterpret_cast<uint32_t*>(
        mpz_export(nullptr, &arraySize, -1, sizeof(uint32_t), 0, 0, number.get_mpz_t())
    ), serializedBigIntDeleter());
    return make_pair(std::move(pointer), arraySize);
}

inline BigInt arrayToBigInt(const SerializedBigInt& data)
{
    BigInt result;
    mpz_import(result.get_mpz_t(), data.second, -1, sizeof(uint32_t), 0, 0, data.first.get());
    return result;
}

QuadraticSieveScheduler::QuadraticSieveScheduler(const std::function<ElfPointer()>& factory) :
    SchedulerTemplate(factory)
{

}

void QuadraticSieveScheduler::provideData(istream& input)
{
    input >> number;

    if (input.fail())
        throw runtime_error("Failed to read BigInt from input stream in " __FILE__);
}

void QuadraticSieveScheduler::outputData(ostream& output)
{
    output << p << endl;
    output << q << endl;
}

void QuadraticSieveScheduler::generateData(const DataGenerationParameters&)
{
    // TODO: Use parameters and generate a random number
    number = 1089911;
}

bool QuadraticSieveScheduler::hasData() const
{
    return number != 0;
}

void QuadraticSieveScheduler::doSimpleDispatch()
{
    tie(p, q) = QuadraticSieveHelper::factor(
        number,
        bind(&QuadraticSieveElf::sieveSmoothSquares, &elf(), _1, _2, _3, _4)
    );
}

void packBigintVector(vector<size_t>& outSizes, vector<uint32_t>& outData, const vector<BigInt>& input)
{
    for (const auto& number : input)
    {
        auto serialized = bigIntToArray(number);
        outSizes.push_back(serialized.second);
        outData.insert(outData.end(), serialized.first.get(), serialized.first.get() + serialized.second);
    }
}

SmoothSquareList unpackBigintVector(const vector<size_t>& sizes, const vector<uint32_t>& data)
{
    SmoothSquareList result;
    auto dataIter = data.begin();
    for (const auto size : sizes)
    {
        SerializedBigInt serialized;
        serialized.second = size;
        serialized.first.reset(reinterpret_cast<uint32_t*>(malloc(sizeof(uint32_t)*size)));
        std::copy(dataIter, dataIter+size, serialized.first.get());
        dataIter += size;
        result.push_back(arrayToBigInt(serialized));
    }

    return result;
}

void QuadraticSieveScheduler::doDispatch()
{
    if (MpiHelper::isMaster())
    {
        tie(p, q) = QuadraticSieveHelper::factor(
            number,
            bind(&QuadraticSieveScheduler::sieveDistributed, this, _1, _2, _3, _4)
        );
    }
    else
    {
        BigInt number = receiveBigInt();
        BigInt start = receiveBigIntFromMaster();
        BigInt end = receiveBigIntFromMaster();
        size_t factorBaseSize;
        MPI::COMM_WORLD.Bcast(&factorBaseSize, 1, MPI::UNSIGNED_LONG, MpiHelper::MASTER);
        vector<smallPrime_t> factorBase(factorBaseSize);
        MPI::COMM_WORLD.Bcast(factorBase.data(), factorBaseSize, MPI::INT, MpiHelper::MASTER);

        auto result = elf().sieveSmoothSquares(start, end, number, factorBase);
        int resultSize = result.size();
        MPI::COMM_WORLD.Gather(&resultSize, 1, MPI::INT, nullptr, 0, MPI::INT, MpiHelper::MASTER);

        vector<size_t> resultsSizes;
        vector<uint32_t> resultsData;
        packBigintVector(resultsSizes, resultsData, result);

        MPI::COMM_WORLD.Gatherv(resultsSizes.data(), resultSize, MPI::UNSIGNED_LONG, nullptr, nullptr, nullptr, MPI::UNSIGNED_LONG, MpiHelper::MASTER);
        MPI::COMM_WORLD.Gatherv(resultsData.data(), resultsData.size(), MPI::INT, nullptr, nullptr, nullptr, MPI::INT, MpiHelper::MASTER);
    }
}


void sendBigInt(const BigInt& number)
{
    auto serialized = bigIntToArray(number);
    MPI::COMM_WORLD.Bcast(&serialized.second, 1, MPI::UNSIGNED_LONG, MpiHelper::MASTER);
    MPI::COMM_WORLD.Bcast(serialized.first.get(), serialized.second, MPI::INT, MpiHelper::MASTER);
}

BigInt receiveBigInt()
{
    SerializedBigInt result;
    MPI::COMM_WORLD.Bcast(&result.second, 1, MPI::UNSIGNED_LONG, MpiHelper::MASTER);
    result.first.reset(reinterpret_cast<uint32_t*>(malloc(result.second * sizeof(uint32_t))));
    MPI::COMM_WORLD.Bcast(result.first.get(), result.second, MPI::INT, MpiHelper::MASTER);
    return arrayToBigInt(result);
}

void sendBigIntTo(const BigInt& number, int nodeId)
{
    auto serialized = bigIntToArray(number);
    MPI::COMM_WORLD.Send(&serialized.second, 1, MPI::UNSIGNED_LONG, nodeId, 0);
    MPI::COMM_WORLD.Send(serialized.first.get(), serialized.second, MPI::INT, nodeId, 0);
}

BigInt receiveBigIntFromMaster()
{
    SerializedBigInt result;
    MPI::COMM_WORLD.Recv(&result.second, 1, MPI::UNSIGNED_LONG, MpiHelper::MASTER, 0);
    result.first.reset(reinterpret_cast<uint32_t*>(malloc(result.second * sizeof(uint32_t))));
    MPI::COMM_WORLD.Recv(result.first.get(), result.second, MPI::INT, MpiHelper::MASTER, 0);
    return arrayToBigInt(result);
}

void QuadraticSieveScheduler::doBenchmarkDispatch(NodeId node)
{
    if (MpiHelper::isMaster())
    {
        BigInt start = 15510279;
        BigInt end = 15557641;
        BigInt number("240568746979579");
        FactorBase factorBase = {
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
            67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137,
            139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211,
            223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283,
            293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379,
            383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461,
            463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563,
            569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643,
            647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739,
            743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829,
            839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937,
            941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031,
            1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117,
            1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229,
            1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327
        };
        sendBigIntTo(start, node);
        sendBigIntTo(end, node);
        sendBigIntTo(number, node);
        size_t factorBaseSize = factorBase.size();
        MPI::COMM_WORLD.Send(&factorBaseSize, 1, MPI::UNSIGNED_LONG, node, 0);
        MPI::COMM_WORLD.Send(factorBase.data(), factorBaseSize, MPI::INT, node, 0);
    }
    else
    {
        BigInt start = receiveBigIntFromMaster();
        BigInt end = receiveBigIntFromMaster();
        BigInt number = receiveBigIntFromMaster();

        size_t factorBaseSize;
        MPI::COMM_WORLD.Recv(&factorBaseSize, 1, MPI::UNSIGNED_LONG, MpiHelper::MASTER, 0);
        FactorBase factorBase(factorBaseSize);
        MPI::COMM_WORLD.Recv(factorBase.data(), factorBaseSize, MPI::INT, MpiHelper::MASTER, 0);

        elf().sieveSmoothSquares(start, end, number, factorBase);
    }
}

vector<BigInt> QuadraticSieveScheduler::sieveDistributed(
    const BigInt& start,
    const BigInt& end,
    const BigInt& number,
    const FactorBase& factorBase
)
{
    sendBigInt(number);

    BigInt intervalLength = end - start;
    BigInt chunkSize = div_ceil(intervalLength, BigInt(MpiHelper::numberOfNodes()));

    for (size_t nodeId=1; nodeId < MpiHelper::numberOfNodes(); ++nodeId)
    {
        BigInt partialStart = start + chunkSize*nodeId;
        BigInt partialEnd = min(partialStart + chunkSize, end);
        sendBigIntTo(partialStart, nodeId);
        sendBigIntTo(partialEnd, nodeId);
    }

    size_t factorBaseSize = factorBase.size();
    MPI::COMM_WORLD.Bcast(&factorBaseSize, 1, MPI::UNSIGNED_LONG, MpiHelper::MASTER);
    MPI::COMM_WORLD.Bcast(const_cast<smallPrime_t*>(factorBase.data()), factorBaseSize, MPI::INT, MpiHelper::MASTER);

    SmoothSquareList masterResult = elf().sieveSmoothSquares(start, min(start + chunkSize, end), number, factorBase);
    size_t resultSize = masterResult.size();

    vector<size_t> numberSizes;
    vector<uint32_t> numberData;
    packBigintVector(numberSizes, numberData, masterResult);

    vector<int> resultSizes(MpiHelper::numberOfNodes());
    MPI::COMM_WORLD.Gather(&resultSize, 1, MPI::INT, resultSizes.data(), 1, MPI::INT, MpiHelper::MASTER);

    auto totalSmoothSquares = std::accumulate(resultSizes.begin(), resultSizes.end(), 0);


    // gather sizes of smooth squares
    vector<int> displs(MpiHelper::numberOfNodes());
    displs[0] = 0;
    for (size_t nodeId = 1; nodeId < MpiHelper::numberOfNodes(); ++nodeId)
    {
        displs[nodeId] = displs[nodeId - 1] + resultSizes.at(nodeId - 1);
    }
    vector<size_t> allNumberSizes(totalSmoothSquares);
    MPI::COMM_WORLD.Gatherv(numberSizes.data(), resultSize, MPI::UNSIGNED_LONG, allNumberSizes.data(), resultSizes.data(), displs.data(), MPI::UNSIGNED_LONG, MpiHelper::MASTER);


    // gather smooth squares
    auto numberSizesSum = std::accumulate(allNumberSizes.begin(), allNumberSizes.end(), 0);
    vector<uint32_t> allNumberData(numberSizesSum);
    vector<int> sizePerNode(MpiHelper::numberOfNodes());
    auto accumulateStart = allNumberSizes.begin();
    //auto accumulateEnd = allNumberSizes.begin() + resultSizes[0];
    for (size_t nodeId = 0; nodeId < MpiHelper::numberOfNodes(); ++nodeId)
    {
        auto accumulateEnd = accumulateStart + resultSizes[nodeId];
        sizePerNode[nodeId] = std::accumulate(accumulateStart, accumulateEnd, 0);
        accumulateStart = accumulateEnd;

    }
    vector<int> dataDispls(MpiHelper::numberOfNodes());
    dataDispls[0] = 0;
    for (size_t nodeId = 1; nodeId < MpiHelper::numberOfNodes(); ++nodeId)
    {
        dataDispls[nodeId] = dataDispls[nodeId - 1] + sizePerNode[nodeId - 1];
    }

    MPI::COMM_WORLD.Gatherv(numberData.data(), numberData.size(), MPI::INT, allNumberData.data(), sizePerNode.data(), dataDispls.data(), MPI::INT, MpiHelper::MASTER);

    return unpackBigintVector(allNumberSizes, allNumberData);
}
