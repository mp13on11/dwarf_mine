#include "QuadraticSieve.h"
#include "QuadraticSieveScheduler.h"
#include "QuadraticSieveElf.h"
#include "common/ProblemStatement.h"
#include <mpi.h>

using namespace std;
using namespace std::placeholders;

vector<BigInt> sieveDistributed(
    const BigInt& start,
    const BigInt& end,
    const BigInt& number,
    const FactorBase& factorBase
);

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
    tie(p, q) = QuadraticSieveHelper::factor(number, sieveDistributed);
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
    vector<BigInt> relations;
    if (MpiHelper::isMaster())
    {
        sendBigInt(start);
        sendBigInt(end);
        sendBigInt(number);

    }
    else
    {
        BigInt start = receiveBigInt();
        BigInt end = receiveBigInt();
        BigInt number = receiveBigInt();
        cout << start << ", " << end << ", " << number << endl;
    }
    return relations;
}
