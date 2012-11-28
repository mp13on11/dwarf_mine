#!/usr/bin/python
# -*- coding: utf8 -*-

# Requires at least Python 2.7

import subprocess, os

THRESHOLD = 1000.0
UNITS = ['µs', 'ms', 's']
INA="big_left.txt"
INB="big_right.txt"
OUT="out.txt"

def humanize(runtime):
    unit = 0
    
    while runtime >= THRESHOLD and unit < 2:
        runtime /= THRESHOLD
        unit += 1
        
    return "time (%s): %f" % (UNITS[unit], runtime)

def invokePlatform(platform):
    command = ["build/src/" + platform, INA, INB, OUT]
    output = subprocess.check_output(command, stderr=subprocess.STDOUT)
    runtime = output.split(':')[1]
    return humanize(float(runtime))

for platform in ("mpi/mpi-matrix", "cuda/cuda", "smp/smp"):
    print platform + " {" 
    print "\t" + invokePlatform(platform)
    print "}"

os.unlink(OUT)
