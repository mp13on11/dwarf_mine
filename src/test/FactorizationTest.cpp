#include "factorize/BigInt.h"
#include "factorize/QuadraticSieve.h"

#include <gtest/gtest.h>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <chrono>
#include <functional>
#include <utility>
#include <bitset>
#include <map>
#include <memory>
#include <algorithm>
#include <random>

using namespace std;
using namespace chrono;
using namespace testing;



TEST(BigIntTest, testFactorizationQuadraticSieve)
{
    //BigInt p("551226983117");
    //BigInt q("554724632351");
    BigInt p("1313839");
    BigInt q("1327901");
    BigInt n = p*q; 


    QuadraticSieve qs(n);

    auto start = high_resolution_clock::now();

    auto pq = qs.factorize();

    auto end = high_resolution_clock::now();
    milliseconds elapsed = duration_cast<milliseconds>(end - start);
    std::cout << "total time: " << elapsed.count() / 1000.0 << " seconds" << endl;  


    cout << "Solution found! :-D" << endl;
    cout << " N  = " << n << endl;
    cout << " p  = " << pq.first << endl;
    cout << " q  = " << pq.second << endl;
    cout << "p*q = " << pq.first * pq.second << endl;
}


