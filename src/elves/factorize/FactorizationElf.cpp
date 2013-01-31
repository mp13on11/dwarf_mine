#include "FactorizationElf.h"
#include "QuadraticSieve.h"

#include <iostream>
#include <stdexcept>
#include <string>

using namespace std;

void FactorizationElf::run(istream& input, ostream& output)
{
    BigInt number;
    input >> number;

    if (input.fail())
        throw runtime_error("Failed to read BigInt from istream in " __FILE__);

    pair<BigInt, BigInt> result = factorize(number);

    output << result.first << endl;
    output << result.second << endl;
}
