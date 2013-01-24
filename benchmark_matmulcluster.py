#!/usr/bin/python
import os
import sys
from math import sqrt
from matplotlib import pyplot
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from datetime import datetime


def timesFromFile(fileName):
    return [float(line) for line in open(fileName)]

def avg(xs):
    return sum(xs) / len(xs)

def plotSpeedUp(threadTimes, fileName):
    xticks = range(1, len(threadTimes)+1)

    alltimes = threadTimes.values()

    oneThreadAvgTime = avg(threadTimes[1])

    allSpeedUps = [[oneThreadAvgTime / t for t in ts] for ts in alltimes]

    avgSpeedUps = [avg(ts) for ts in allSpeedUps]

    fig = pyplot.figure(figsize=(20, 10))

    pyplot.plot(xticks, avgSpeedUps, '-', c="grey")


    pyplot.boxplot(allSpeedUps)

    pyplot.xticks(xticks, threadTimes.keys())

    ax = fig.add_subplot(1,1,1)

    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))

    ax.xaxis.grid(True,'minor',linestyle='-', color='0.95', linewidth=1)
    ax.yaxis.grid(True,'minor',linestyle='-', color='0.95', linewidth=1)

    ax.xaxis.grid(True,'major',linestyle='-', color='0.5', linewidth=1)


    ax.set_axisbelow(True)

    pyplot.ylim(0)

    pyplot.title('MatMul (SMP, 1000x1000)')
    pyplot.xlabel('OMP_NUM_THREADS')
    pyplot.ylabel('SpeedUp')
    pyplot.show()
    pyplot.savefig(fileName, dpi=80)


def plotBurnDown(threadTimes, fileName):
    xticks = range(1, len(threadTimes)+1)

    alltimes = threadTimes.values()

    oneThreadAvgTime = avg(threadTimes[1])

    allSpeedUps = [[oneThreadAvgTime / t / threads for t in ts] for threads, ts in threadTimes.items()]

    avgSpeedUps = [avg(ts) for ts in allSpeedUps]

    fig = pyplot.figure(figsize=(20, 10))

    pyplot.plot(xticks, avgSpeedUps, '-', c="grey")


    pyplot.boxplot(allSpeedUps)

    pyplot.xticks(xticks, threadTimes.keys())

    ax = fig.add_subplot(1,1,1)

    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax.xaxis.grid(True,'minor',linestyle='-', color='0.95', linewidth=1)
    ax.yaxis.grid(True,'minor',linestyle='-', color='0.95', linewidth=1)

    ax.xaxis.grid(True,'major',linestyle='-', color='0.5', linewidth=1)


    ax.set_axisbelow(True)

    pyplot.ylim(0)

    pyplot.title('MatMul (SMP, 1000x1000)')
    pyplot.xlabel('OMP_NUM_THREADS')
    pyplot.ylabel('Burndown (SpeedUp / Thread)')
    pyplot.show()
    pyplot.savefig(fileName, dpi=80)



file_dir = sys.argv[1]
iterations = int(sys.argv[2]) if (len(sys.argv)>2) else 100
warmups = iterations / 10
matrixSize = 1000


def timeFile(smpHosts, cudaHosts, ext = ".txt"):
    hostsString = ",".join(smpHosts) + "-" + ",".join(cudaHosts)
    return os.path.join(file_dir, "matmul_cluster_{0}_{1}{2}".format(hostsString, matrixSize, ext))

def commandLine(smpHosts, cudaHosts):
    hostLine =  "-host {0} -np 1 ./build/src/main/dwarf_mine -m {1} -c matrix -w {3} -n {4} --time_output {5} "\
                "--left_rows {2} --common_rows_columns {2} --right_columns {2}"

    timeFileName = timeFile(smpHosts, cudaHosts)
    smpLines = [hostLine.format(host, "smp", matrixSize, warmups, iterations, timeFileName) for host in smpHosts]
    cudaLines = [hostLine.format(host, "cuda", matrixSize, warmups, iterations, timeFileName) for host in cudaHosts]

    mpirunArgs = " : ".join(smpLines + cudaLines)

    return "mpirun --tag-output %s" % mpirunArgs


def main():
    q1, q2, q3 = ("quadcore1", "quadcore2", "quadcore3")
    scenarios = [((q3,), ()), ((q3,q2), ()), ((q3,q2,q1), ())]

    numberOfScenarios = range(1, len(scenarios)+2)

    for smpHosts, cudaHosts in scenarios:
        print "Executing on smpHosts:", smpHosts, "and cudaHosts:", cudaHosts
        print "commandLine:", commandLine(smpHosts, cudaHosts)
        startTime = datetime.now();
        os.system(commandLine(smpHosts, cudaHosts))
        print "elapsed:", (datetime.now() - startTime).total_seconds(), "s"

    alltimes = dict((len(smpHosts)+len(cudaHosts), timesFromFile(timeFile(smpHosts, cudaHosts))) for smpHosts, cudaHosts in scenarios)

    print "Plotting..."
    plotSpeedUp(alltimes, timeFile("", "", "_speedup.png"))
    plotBurnDown(alltimes, timeFile("", "", "_burndown.png"))


main()