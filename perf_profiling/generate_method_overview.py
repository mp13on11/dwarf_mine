#!/usr/bin/python
# Usage: python generate_method_overview.py path/to/perf.data [-v]

import sys, subprocess, pprint

tempFile = ".generate_method_overview.tmp"
totalKey = "__TOTAL"
isVerbose = False
eventMethodCounts = {}

def main(argv):
    global isVerbose
    isVerbose = len(argv) == 2 and argv[1] == "-v"
    command = commandLine(argv[0])
    executeCommand(command)
    parseResults()
    cleanUp()
    plot()
    #if isVerbose:
    #    printSummary()

def executeCommand(command):
    if isVerbose:
        print "Executing \"" + command + "\""
    subprocess.call(command, shell=True)
    if isVerbose:
        print "Done executing"

def parseResults():
    if isVerbose:
        print "Parsing " + tempFile
    with open(tempFile, 'rU') as file:
        for line in file:
            parseLine(line)
    determineTotals()
    if isVerbose:
        print "Done parsing"

def parseLine(tempFileLine):
    # Example line:
    # dwarf_mine  3868 6148526.297100: cache-misses:      7faf6895d21a Matrix<float>::columns() const (/home/mp13on11/henning/dwarf_mine/build/src/elves/matrix/libmatrix.so)
    elements = tempFileLine.split()
    # Binary = elements[0]
    # PID (?) = elements[1]
    time = elements[2][0:-1]            # Cut ":"
    event = elements[3][0:-1]           # Cut ":"
    # Function address = elements[4]
    method = " ".join(elements[5:-1])   # E.g. concat ["Matrix<float>::columns()", "const"]
    # Lib file = elements[-1]
    adoptEventMethodCount(event, time, method)

def adoptEventMethodCount(event, time, method):
    global eventMethodCounts
    if not eventMethodCounts.has_key(event):
        eventMethodCounts[event] = {}
    if not eventMethodCounts[event].has_key(method):
        eventMethodCounts[event][method] = []
    eventMethodCounts[event][method].append(time)

def determineTotals():
    for event in eventMethodCounts.keys():
        methods = len(eventMethodCounts[event].keys())
        total = 0
        for method in eventMethodCounts[event]:
            total += len(eventMethodCounts[event][method])
        eventMethodCounts[event][totalKey] = total

def plot():
    print ""
    for event in eventMethodCounts.keys():
        methodValues = []
        total = eventMethodCounts[event][totalKey]
        print "Overview for event \"" + event + "\" (" + str(total) + "):"
        for method in eventMethodCounts[event].keys():
            if method == totalKey:
                continue
            methodTotal = len(eventMethodCounts[event][method])
            ratio = round(100 * ((0.0 + methodTotal)/total), 2)
            methodValues.append((method, methodTotal, ratio))
        for values in sorted(methodValues, key=lambda values: values[2], reverse=True):
            print "\t" + str(values[1]) + "\t" + str(values[2]) + "\t" + values[0]
        print ""

def commandLine(perfDataFile):
    return "perf script -G -i " + perfDataFile + " | grep dwarf_mine\/build > " + tempFile

def cleanUp():
    subprocess.call("rm " + tempFile, shell=True)
    if isVerbose:
        print "Removed " + tempFile

def printSummary():
    print "=== SUMMARY ==="
    for event in eventMethodCounts.keys():
        methods = len(eventMethodCounts[event].keys()) - 1 # total
        print event + ": " + str(eventMethodCounts[event][totalKey]) + " in " + str(methods) +" methods"


if __name__ == "__main__":
    main(sys.argv[1:])
