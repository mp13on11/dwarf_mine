#include "SimpleMonteCarloScheduler.h"
#include "montecarlo/MonteCarloElf.h"
#include <iostream>
#include "montecarlo/OthelloUtil.h"

using namespace std;

SimpleMonteCarloScheduler::SimpleMonteCarloScheduler(const function<ElfPointer()>& factory) :
        MonteCarloScheduler(factory)
{
}

void SimpleMonteCarloScheduler::doDispatch()
{
    _result = elf().getBestMoveFor(_state, _repetitions);
    cout << "{" << _result.x << ", " << _result.y << "} " << _result.wins << "/" << _result.visits << " "<< _result.wins * 1.0 / _result.visits << " (" << _result.iterations<<")"<<endl;
    
}
