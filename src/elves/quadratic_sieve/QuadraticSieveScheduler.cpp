#include "QuadraticSieve.h"
#include "QuadraticSieveScheduler.h"
#include "QuadraticSieveElf.h"
#include "common/ProblemStatement.h"
#include <mpi.h>

std::ostream& operator<<(std::ostream& stream, const std::vector<size_t>& list)
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

vector<BigInt> sieveDistributed(
    const BigInt& start,
    const BigInt& end,
    const BigInt& number,
    const FactorBase& factorBase
);

BigInt receiveBigInt();

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

void QuadraticSieveScheduler::doDispatch()
{
    cout << "rank: " << MpiHelper::rank() << endl;

    if (MpiHelper::isMaster())
    {
        tie(p, q) = QuadraticSieveHelper::factor(number, sieveDistributed);
    }
    else
    {
        cout << "Slave" << endl;
        BigInt start = receiveBigInt();
        BigInt end = receiveBigInt();
        BigInt number = receiveBigInt();
        size_t factorBaseSize;
        MPI::COMM_WORLD.Bcast(&factorBaseSize, 1, MPI::UNSIGNED_LONG, MpiHelper::MASTER);
        vector<smallPrime_t> factorBase(factorBaseSize);
        MPI::COMM_WORLD.Bcast(factorBase.data(), factorBaseSize, MPI::INT, MpiHelper::MASTER);
        cout << start << ", " << end << ", " << number << ", " << factorBaseSize << endl;

        auto result = elf().sieveSmoothSquares(start, end, number, factorBase);
        auto resultSize = result.size();
        cout << resultSize << endl;
        MPI::COMM_WORLD.Gather(&resultSize, 1, MPI::UNSIGNED_LONG, nullptr, 0, MPI::UNSIGNED_LONG, MpiHelper::MASTER);
    }
}

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

vector<BigInt> sieveDistributed(
    const BigInt& start,
    const BigInt& end,
    const BigInt& number,
    const FactorBase& factorBase
)
{
    vector<BigInt> smooths;
    sendBigInt(start);
    sendBigInt(end);
    sendBigInt(number);
    size_t factorBaseSize = factorBase.size();
    MPI::COMM_WORLD.Bcast(&factorBaseSize, 1, MPI::UNSIGNED_LONG, MpiHelper::MASTER);
    MPI::COMM_WORLD.Bcast(const_cast<smallPrime_t*>(factorBase.data()), factorBaseSize, MPI::INT, MpiHelper::MASTER);

    size_t bla = 42;
    vector<size_t> resultSizes(MpiHelper::numberOfNodes());
    MPI::COMM_WORLD.Gather(&bla, 1, MPI::UNSIGNED_LONG, resultSizes.data(), 1, MPI::UNSIGNED_LONG, MpiHelper::MASTER);
    cout << resultSizes << endl;

    return smooths;
}
