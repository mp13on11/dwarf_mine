#pragma once

#include <map>
#include <string>
#include <functional>
#include <memory>

class MatrixOnlineSchedulingStrategy;

class MatrixOnlineSchedulingStrategyFactory
{
public:
    static std::unique_ptr<MatrixOnlineSchedulingStrategy> getStrategy(const std::string& strategy);

private:
    static std::map<std::string, std::function<std::unique_ptr<MatrixOnlineSchedulingStrategy>()>> strategies;
    template <typename T>
    static std::unique_ptr<MatrixOnlineSchedulingStrategy> getStrategy();
};
