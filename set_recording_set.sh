#!/bin/bash

usage()
{
echo "Usage: . set_recording_set.sh option

WARNING: You MUST type the . character to allow the script to set global environment variables.

Sets global environment variables for VampirTrace to record corresponding performance events.

OPTIONS:
    cache_l1
    cache_l2
    cache_l3
    branching
    flop"
}

setRecordingSet()
{
    export VT_METRICS=$1
}

export VT_MODE=STAT:TRACE
export VT_STAT_PROPS=ALL
export VT_MAX_FLUSHES=0

case $1 in
    cache_l1)
        echo "Recording L1 data cache misses, loads, stores and instruction cache misses."
        setRecordingSet "PAPI_L1_DCM:PAPI_L1_LDM:PAPI_L1_STM:PAPI_L1_ICM"
        ;;
    cache_l2)
        echo "Recording L2 data cache misses, stores and instruction cache accesses, misses."
        setRecordingSet "PAPI_L2_DCM:PAPI_L2_STM:PAPI_L2_ICA:PAPI_L2_ICM"
        ;;
    cache_l3)
        echo "Recording L3 cache accesses and misses."
        setRecordingSet "PAPI_L3_TCA:PAPI_L3_TCM"
        ;;
    branching)
        echo "Recording correctly predicted and mispredicted branchings."
        setRecordingSet "PAPI_BR_PRC:PAPI_BR_MSP"
        ;;
    flop)
        # http://icl.cs.utk.edu/projects/papi/wiki/PAPITopics:SandyFlops
        echo "Recording double precision floating point scalar and vector operations."
        setRecordingSet "PAPI_DP_OPS"
        ;;
    *)
        usage
        ;;
esac
