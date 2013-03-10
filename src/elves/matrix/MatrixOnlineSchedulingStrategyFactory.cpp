#include "MatrixOnlineSchedulingStrategyFactory.h"
#include "MatrixOnlineSchedulingRowwise.h"

using namespace std;

map<string, function<unique_ptr<MatrixOnlineSchedulingStrategy>()>> MatrixOnlineSchedulingStrategyFactory::strategies =
{
    {
        "row-wise",
        &getStrategy<MatrixOnlineSchedulingRowwise>
    }
};

vector<string> MatrixOnlineSchedulingStrategyFactory::getStrategies()
{
    vector<string> strategiesList;
    for (const auto& strategyDefinition : strategies)
        strategiesList.push_back(strategyDefinition.first);
    return strategiesList;
}
    
unique_ptr<MatrixOnlineSchedulingStrategy>
MatrixOnlineSchedulingStrategyFactory::getStrategy(const std::string& strategy)
{
    if (strategies.find(strategy) == strategies.end())
        throw "ERROR: Can't find desired matrix online scheduling mode.";
    return strategies[strategy]();
}

template <typename T>
unique_ptr<MatrixOnlineSchedulingStrategy> 
MatrixOnlineSchedulingStrategyFactory::getStrategy()
{
    return unique_ptr<T>(new T());
}

