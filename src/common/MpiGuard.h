#pragma once

class Configuration;

class MpiGuard
{
public:
    MpiGuard(const Configuration& configuration, int argc, char** argv);
    ~MpiGuard();

    MpiGuard(const MpiGuard& copy) = delete;
    MpiGuard(MpiGuard&& move) = delete;
    MpiGuard& operator=(const MpiGuard& rhs) = delete;
    MpiGuard& operator=(const MpiGuard&& rhs) = delete;

    static int getThreadSupport();

protected:
    static int threadSupport;
};
