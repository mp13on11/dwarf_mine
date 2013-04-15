#include "QuadraticSieveScheduler.h"
#include "QuadraticSieveElf.h"
#include "common/ProblemStatement.h"
#include "common/Utils.h"
#include <numeric>

using namespace std;
using namespace std::placeholders;

BigInt receiveBigInt(const Communicator& communicator);
BigInt receiveBigIntFromMaster(const Communicator& communicator);

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

QuadraticSieveScheduler::QuadraticSieveScheduler(const Communicator& communicator, const std::function<ElfPointer()>& factory) :
    SchedulerTemplate(communicator, factory)
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
    number = BigInt("694913266134731001458300843412962010901");
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
    if (communicator.isMaster())
    {
        tie(p, q) = QuadraticSieveHelper::factor(
            number,
            bind(&QuadraticSieveScheduler::sieveDistributed, this, _1, _2, _3, _4)
        );
    }
    else
    {
        BigInt number = receiveBigInt(communicator);
        BigInt start = receiveBigIntFromMaster(communicator);
        BigInt end = receiveBigIntFromMaster(communicator);

        size_t factorBaseSize;
        communicator->Bcast(&factorBaseSize, 1, MPI::UNSIGNED_LONG, Communicator::MASTER_RANK);
        vector<smallPrime_t> factorBase(factorBaseSize);
        communicator->Bcast(factorBase.data(), factorBaseSize, MPI::INT, Communicator::MASTER_RANK);

        auto result = elf().sieveSmoothSquares(start, end, number, factorBase);
        int resultSize = result.size();
        communicator->Gather(&resultSize, 1, MPI::INT, nullptr, 0, MPI::INT, Communicator::MASTER_RANK);

        vector<size_t> resultsSizes;
        vector<uint32_t> resultsData;
        packBigintVector(resultsSizes, resultsData, result);

        communicator->Gatherv(resultsSizes.data(), resultSize, MPI::UNSIGNED_LONG, nullptr, nullptr, nullptr, MPI::UNSIGNED_LONG, Communicator::MASTER_RANK);
        communicator->Gatherv(resultsData.data(), resultsData.size(), MPI::INT, nullptr, nullptr, nullptr, MPI::INT, Communicator::MASTER_RANK);
    }
}


void sendBigInt(const Communicator& communicator, const BigInt& number)
{
    auto serialized = bigIntToArray(number);
    communicator->Bcast(&serialized.second, 1, MPI::UNSIGNED_LONG, Communicator::MASTER_RANK);
    communicator->Bcast(serialized.first.get(), serialized.second, MPI::INT, Communicator::MASTER_RANK);
}

BigInt receiveBigInt(const Communicator& communicator)
{
    SerializedBigInt result;
    communicator->Bcast(&result.second, 1, MPI::UNSIGNED_LONG, Communicator::MASTER_RANK);
    result.first.reset(reinterpret_cast<uint32_t*>(malloc(result.second * sizeof(uint32_t))));
    communicator->Bcast(result.first.get(), result.second, MPI::INT, Communicator::MASTER_RANK);
    return arrayToBigInt(result);
}

void sendBigIntTo(const Communicator& communicator, const BigInt& number, int nodeId)
{
    auto serialized = bigIntToArray(number);
    communicator->Send(&serialized.second, 1, MPI::UNSIGNED_LONG, nodeId, 0);
    communicator->Send(serialized.first.get(), serialized.second, MPI::INT, nodeId, 0);
}

BigInt receiveBigIntFromMaster(const Communicator& communicator)
{
    SerializedBigInt result;
    communicator->Recv(&result.second, 1, MPI::UNSIGNED_LONG, Communicator::MASTER_RANK, 0);
    result.first.reset(reinterpret_cast<uint32_t*>(malloc(result.second * sizeof(uint32_t))));
    communicator->Recv(result.first.get(), result.second, MPI::INT, Communicator::MASTER_RANK, 0);
    return arrayToBigInt(result);
}

vector<double> QuadraticSieveScheduler::determineChunkSizes(const BigInt& start, const BigInt& end)
{
    BigInt intervalLengthBigInt = end - start;

    if (!intervalLengthBigInt.fits_ulong_p())
        throw runtime_error("Interval is too long!");

    uint64_t intervalLength = intervalLengthBigInt.get_ui();
    auto weights = communicator.weights();

    vector<double> chunkSizes;
    for (auto weight : weights)
        chunkSizes.push_back(intervalLength * weight);

    return chunkSizes;
}

vector<BigInt> QuadraticSieveScheduler::sieveDistributed(
    const BigInt& start,
    const BigInt& end,
    const BigInt& number,
    const FactorBase& factorBase
)
{
    sendBigInt(communicator, number);

    auto chunkSizes = determineChunkSizes(start, end);

    for (size_t nodeId=1; nodeId < communicator.size(); ++nodeId)
    {
        double weightedChunkSize = chunkSizes[nodeId];
        BigInt partialStart = start + weightedChunkSize*nodeId;
        BigInt partialEnd = min(partialStart + weightedChunkSize, end);
        sendBigIntTo(communicator, partialStart, nodeId);
        sendBigIntTo(communicator, partialEnd, nodeId);
    }

    size_t factorBaseSize = factorBase.size();
    communicator->Bcast(&factorBaseSize, 1, MPI::UNSIGNED_LONG, Communicator::MASTER_RANK);
    communicator->Bcast(const_cast<smallPrime_t*>(factorBase.data()), factorBaseSize, MPI::INT, Communicator::MASTER_RANK);

    double masterChunkSize = chunkSizes[0];
    SmoothSquareList masterResult;

    if (masterChunkSize > 0.0) 
        masterResult  = elf().sieveSmoothSquares(start, min(start + masterChunkSize, end), number, factorBase);
    size_t resultSize = masterResult.size();

    vector<size_t> numberSizes;
    vector<uint32_t> numberData;
    packBigintVector(numberSizes, numberData, masterResult);

    vector<int> resultSizes(communicator.size());
    communicator->Gather(&resultSize, 1, MPI::INT, resultSizes.data(), 1, MPI::INT, Communicator::MASTER_RANK);

    auto totalSmoothSquares = std::accumulate(resultSizes.begin(), resultSizes.end(), 0);


    // gather sizes of smooth squares
    vector<int> displs(communicator.size());
    displs[0] = 0;
    for (size_t nodeId = 1; nodeId < communicator.size(); ++nodeId)
    {
        displs[nodeId] = displs[nodeId - 1] + resultSizes.at(nodeId - 1);
    }
    vector<size_t> allNumberSizes(totalSmoothSquares);
    communicator->Gatherv(numberSizes.data(), resultSize, MPI::UNSIGNED_LONG, allNumberSizes.data(), resultSizes.data(), displs.data(), MPI::UNSIGNED_LONG, Communicator::MASTER_RANK);


    // gather smooth squares
    auto numberSizesSum = std::accumulate(allNumberSizes.begin(), allNumberSizes.end(), 0);
    vector<uint32_t> allNumberData(numberSizesSum);
    vector<int> sizePerNode(communicator.size());
    auto accumulateStart = allNumberSizes.begin();
    //auto accumulateEnd = allNumberSizes.begin() + resultSizes[0];
    for (size_t nodeId = 0; nodeId < communicator.size(); ++nodeId)
    {
        auto accumulateEnd = accumulateStart + resultSizes[nodeId];
        sizePerNode[nodeId] = std::accumulate(accumulateStart, accumulateEnd, 0);
        accumulateStart = accumulateEnd;

    }
    vector<int> dataDispls(communicator.size());
    dataDispls[0] = 0;
    for (size_t nodeId = 1; nodeId < communicator.size(); ++nodeId)
    {
        dataDispls[nodeId] = dataDispls[nodeId - 1] + sizePerNode[nodeId - 1];
    }

    communicator->Gatherv(numberData.data(), numberData.size(), MPI::INT, allNumberData.data(), sizePerNode.data(), dataDispls.data(), MPI::INT, Communicator::MASTER_RANK);

    auto unpacked = unpackBigintVector(allNumberSizes, allNumberData);
    return unpacked;
}
