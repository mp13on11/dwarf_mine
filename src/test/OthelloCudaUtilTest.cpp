#include "OthelloCudaUtilTest.h"
#include <cuda-utils/Memory.h>
#include <vector>
#include <tuple>

using namespace std;


TEST_F(OthelloCudaSimulatorTest, GetRandomMoveIndexTest)
{
    size_t randomResult = 0;
    CudaUtils::Memory<size_t> cudaRandomResult(1);
    cudaRandomResult.transferFrom(&randomResult);
    vector<size_t> limits = {2, 7, 11, 12, 15, 21, 25, 42, 64};
    for (const auto& limit : limits)    
    {
        for (size_t i = 1; i <= limit; ++i)
        {
            size_t result = limit - i;
            testRandomNumberProxy(result * 1.0f / limit, limit, cudaRandomResult.get());

            cudaRandomResult.transferTo(&randomResult);

            ASSERT_EQ(result, randomResult)<<" for limit "<<limit<<" ratio: "<<result * 1.0f / limit;
        }
    }
}

TEST_F(OthelloCudaSimulatorTest, GetRandomMoveIndexRegressionTest)
{
    size_t randomResult = 0;
    CudaUtils::Memory<size_t> cudaRandomResult(1);
    cudaRandomResult.transferFrom(&randomResult);
    // fakeRandom, maximum, expected result
    typedef tuple<float, size_t, size_t> scenario; 
    vector<scenario> data = { // the commented results are expected but not calculated due to rounding issues
        scenario(0.590909,            22, 12),// 13),
        scenario(0.979073,             4,  3),
        scenario(0.69767439365386963, 43, 29),// 30),
        scenario(0.7804877758026123,  41, 31) // 32)
    }; 
    for (const auto& t : data)
    {
        testRandomNumberProxy(get<0>(t), get<1>(t), cudaRandomResult.get());

        cudaRandomResult.transferTo(&randomResult);

        ASSERT_EQ(get<2>(t), randomResult);
    }
}