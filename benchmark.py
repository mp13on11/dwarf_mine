#!/usr/bin/python
# -*- coding: utf8 -*-

# Requires at least Python 2.7

import subprocess, os, sys
#import cairoplot

THRESHOLD = 1000.0
UNITS = ['µs', 'ms', 's']
INA="big_left.txt"
INB="big_right.txt"
INhA="huge_left.txt"
INhB="huge_right.txt"
INC="matrix.txt"
OUT="out.txt"
SCENARIOS = ["5x5", "50x50", "500x500"]
PARAMETERS = [(INC, INC, OUT),(INA, INB, OUT),(INhA, INhB, OUT)]

doHumanize = True
plot = False

for arg in sys.argv:
    if arg in ("-h", "--humanize"):
        doHumanize = True
    if arg in ("-p", "--plot"):
        plot = True

def humanize(runtime):
    unit = 0
    
    while runtime >= THRESHOLD and unit < 2:
        runtime /= THRESHOLD
        unit += 1
        
    return "time: %8.3f %s" % (runtime, UNITS[unit])

def invokePlatform(platform, scenarioIndex):
    params = PARAMETERS[scenarioIndex]
    command = ["build/src/" + platform, "--iterations", "1000", params[0], params[1], params[2]]
    output = subprocess.check_output(command, stderr=subprocess.STDOUT)
    runtime = output.split(':')[1]
    return float(runtime)

values = {}
for platform in ("cuda/cuda", "smp/smp", "own/own", "mpi/mpi-matrix"):

    values[platform] = []
    print platform
    for scenarioIndex in range(len(SCENARIOS)):        

        runtime = invokePlatform(platform, scenarioIndex)
        values[platform].append(float(runtime)) 

        if doHumanize:
            runtime = humanize(float(runtime))
        else:
            runtime = "time: %8.3f %s" % (runtime, UNITS[0])
        print "\t", SCENARIOS[scenarioIndex], "\t", (runtime + "")

#if plot:
#    cairoplot.dot_line_plot('Benchmark', values, 400, 300, series_legend = True, axis = True, grid = True, x_labels = SCENARIOS)  

os.unlink(OUT)
