#!/usr/bin/python
import os
import sys
from math import sqrt
from matplotlib import pyplot
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from pylab import *
def timesFromFile(fileName):
    return [float(line) for line in open(fileName)]

def avg(xs):
    return sum(xs) / len(xs)

def chunk(list,size):
    return [list[i:i + size] for i in range(0, len(list), size)]

def plotChange(threadTimes, fileName, iterations):
    chunksPerThread = dict((singleThreadTimes, chunk(threadTimes[singleThreadTimes], iterations)) for singleThreadTimes in threadTimes)
    fig = pyplot.figure(figsize=(20, 10))

    pyplot.title('Othello')
    pyplot.xlabel('Run')
    pyplot.ylabel('Runtime')

    for key in chunksPerThread:
        threadChunk = chunksPerThread[key]

        avgSpeedUps = [[avg(runTimes) / 1000] for runTimes in threadChunk]

        xticks = range(1, len(threadChunk)+1)
        pyplot.xticks(xticks)

        pyplot.plot(xticks, avgSpeedUps, '-')


    pyplot.legend(["#Thread {0}".format(key) for key in chunksPerThread.keys()])
    ylim(0)
    xlim(1)
    # pyplot.show()

    pyplot.savefig(fileName, dpi=80)

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

    pyplot.title('Othello')
    pyplot.xlabel('OMP_NUM_THREADS')
    pyplot.ylabel('SpeedUp')
    #pyplot.show()
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

    pyplot.title('Othello')
    pyplot.xlabel('OMP_NUM_THREADS')
    pyplot.ylabel('Burndown (SpeedUp / Thread)')
    #pyplot.show()
    pyplot.savefig(fileName, dpi=80)



file_dir = sys.argv[1]
iterations = int(sys.argv[2]) if (len(sys.argv)>2) else 100
warmups = iterations / 10


def timeFile(threads, ext = ".txt"):
    return os.path.join(file_dir, "othello_smp_{0}{1}".format(threads, ext))

def commandLine(threads):
    return  "OMP_NUM_THREADS={0} ./build/src/main/dwarf_mine --no_mpi "\
            "-c montecarlo_tree_search "\
            "-m smp -i othello_field -o /dev/null " \
            "-w {1} -n {2} --time_output {3}".format(threads, warmups, iterations, timeFile(threads))


def main():
    numberOfThreads = range(1, 5, 1)

    for threads in numberOfThreads:
        print "Executing with", threads, "thread(s)"
        os.system(commandLine(threads))

    alltimes = dict((threads, timesFromFile(timeFile(threads))) for threads in numberOfThreads)

    print "Plotting..."
    plotSpeedUp(alltimes, timeFile("all", "_speedup.png"))
    plotBurnDown(alltimes, timeFile("all", "_burndown.png"))
    plotChange(alltimes, timeFile("all", "_change.png"), iterations)

main()
