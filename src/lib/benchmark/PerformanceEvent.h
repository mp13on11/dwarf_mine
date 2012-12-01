#pragma once

#include <string>
#include <vector>

class PerformanceEvent
{
public:
    PerformanceEvent(const std::string& name);
    PerformanceEvent(const std::string& name, int code);
    ~PerformanceEvent();
    const std::string& getName() const;
    int getCode() const;

private:
    std::string name;
    mutable int code;
    mutable bool generatedCode;
    void generateCode() const;
};
