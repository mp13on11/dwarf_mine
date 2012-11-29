#pragma once

#include <string>
#include <vector>

class PerformanceEvent
{
private:
    std::string name;
    int code;
    bool generatedCode;
    void generateCode();
    char* getNameAsCharArray() const;
public:
    PerformanceEvent(const std::string& name);
    PerformanceEvent(const std::string& name, const int code);
    ~PerformanceEvent();
    const std::string& getName() const;
    int getCode();
};
