#pragma once

class Profiler
{
public:
	virtual ~Profiler() = 0;

	virtual void beginIterationBlock() = 0;
	virtual void beginIteration() = 0;
	virtual void endIteration() = 0;
	virtual void endIterationBlock() = 0;
};

inline Profiler::~Profiler()
{
}