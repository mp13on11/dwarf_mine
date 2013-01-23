#!/usr/bin/python
import os
import sys
from math import sqrt
from matplotlib import pyplot
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


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


def timeFile(threads, ext = ".txt"):
    return os.path.join(file_dir, "matmul_smp_{0}_1000{1}".format(threads, ext))

def commandLine(threads):
    return "OMP_NUM_THREADS=%i ./build/src/runner/elf_runner -m smp --left_rows 1000 --common_rows_columns 1000 --right_columns 1000 -w 10 -n 100 --time_output %s" % (threads, timeFile(threads))


def main():
    numberOfThreads = range(1, 65, 1)

    for threads in numberOfThreads:
        os.system(commandLine(threads))

    alltimes = dict((threads, timesFromFile(timeFile(threads))) for threads in numberOfThreads)

    plotSpeedUp(alltimes, timeFile("all", "_speedup.png"))
    plotBurnDown(alltimes, timeFile("all", "_burndown.png"))


main()