#!/bin/bash

usage()
{
echo "Usage: . set_recording_set.sh option [buffersize=32]

WARNING: You MUST type the . character to allow the script to set global environment variables.

Sets global environment variables for VampirTrace to record corresponding performance events.
You can specify a per-process buffer size in MB to reduce the amount of flushes. (Default: 32 MB)

OPTIONS:
    none
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
export VT_GNU_NMFILE=/tmp/dwarf_mine.symbols

if [ "$2" != "" ]; then
    echo "Setting VampirTrace buffer size to $2 MB."
    export VT_BUFFER_SIZE="$2M"
else
    # Default to VampirTrace's default (currently 32 MB)
    echo "Defaulting VampirTrace buffer size to 32 MB."
    export VT_BUFFER_SIZE=
fi

case $1 in
    none)
        echo "Not recording performance events."
        setRecordingSet ""
        ;;
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
        echo "Recording double precision floating point scalar and vector operations."
        echo "Please also see http://icl.cs.utk.edu/projects/papi/wiki/PAPITopics:SandyFlops."
        setRecordingSet "PAPI_DP_OPS"
        ;;
    *)
        usage
        ;;
esac
