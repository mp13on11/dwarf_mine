#pragma once

class MpiGuard
{
public:
    MpiGuard(int argc, char** argv);
    ~MpiGuard();

    MpiGuard(const MpiGuard& copy) = delete;
    MpiGuard(MpiGuard&& move) = delete;
    MpiGuard& operator=(const MpiGuard& rhs) = delete;
    MpiGuard& operator=(const MpiGuard&& rhs) = delete;

    static int getThreadSupport();

protected:
    static int threadSupport;
};
