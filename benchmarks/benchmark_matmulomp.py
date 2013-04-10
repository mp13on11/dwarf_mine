#!/usr/bin/python
import os
import sys
from math import sqrt
from matplotlib import pyplot
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def times_from_file(fileName):
    return [float(line) for line in open(fileName)]

def avg(xs):
    return sum(xs) / len(xs)


def plot_speedup(tread_times, file_name):
    all_times = tread_times.values()
    single_process_average_time = avg(tread_times[1])
    all_speedups = [[single_process_average_time / t for t in ts] for ts in all_times]

    plot(all_speedups, 'Speedup', file_name)


def plot_burndown(tread_times, file_name):
    all_times = tread_times.values()
    single_process_average_time = avg(tread_times[1])
    all_burndowns = [[single_process_average_time / t / processes for t in ts] for processes, ts in tread_times.items()]

    plot(all_burndowns, 'Burndown (Speedup / Threads)', file_name)


def plot(values, yaxis_label, file_name):
    figure = pyplot.figure(figsize=(20, 10))

    pyplot.xticks(range(1, len(values)))
    pyplot.boxplot(values)

    set_axes_on(figure)

    pyplot.ylim(ymin=0)

    pyplot.title('MatMul (SMP, 1000x1000)')
    pyplot.xlabel('OMP_NUM_THREADS')
    pyplot.ylabel(yaxis_label)
    pyplot.show()

    pyplot.savefig(file_name, dpi=80)


def set_axes_on(figure):
    axes = figure.add_subplot(1,1,1)

    axes.xaxis.set_major_locator(MultipleLocator(4))
    axes.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    axes.xaxis.set_minor_locator(MultipleLocator(1))
    axes.xaxis.grid(True, 'major',linestyle='-', color='0.5', linewidth=1)
    axes.xaxis.grid(True, 'minor',linestyle='-', color='0.95', linewidth=1)

    axes.yaxis.set_major_locator(MultipleLocator(2))
    axes.yaxis.set_minor_locator(MultipleLocator(0.5))
    axes.yaxis.grid(True, 'minor',linestyle='-', color='0.95', linewidth=1)

    axes.set_axisbelow(True)



file_dir = sys.argv[1]
iterations = int(sys.argv[2]) if (len(sys.argv)>2) else 100
warmups = iterations / 10


def time_file(threads, ext = ".txt"):
    return os.path.join(file_dir, "matmul_smp_{0}_1000{1}".format(threads, ext))

def command_line(threads):
    return  "OMP_NUM_THREADS={0} ./build/src/main/dwarf_mine "\
            "-m smp --left_rows 1000 --common_rows_columns 1000 --right_columns 1000 "\
            "-w {1} -n {2} --time_output {3}".format(threads, warmups, iterations, time_file(threads))


def main():
    numberOfThreads = range(1, 151, 1)

    for threads in numberOfThreads:
        print "Executing with", threads, "thread(s)"
        os.system(command_line(threads))

    alltimes = dict((threads, times_from_file(time_file(threads))) for threads in numberOfThreads)

    print "Plotting..."
    plot_speedup(alltimes, time_file("all", "_speedup.png"))
    plot_burndown(alltimes, time_file("all", "_burndown.png"))


main()