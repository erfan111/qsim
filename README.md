# qsim: A discrete-event simulator for queuing systems

## Requirements
- Python 3.4+
- scipy
- numpy

The simulator uses multiprocessing to speed up the simulation, however that means it requires a considerable amount of memory. Modify the `MULTIPROCESSING_CPUS` accordingly.

## Usage
```
    simulator.py [-h] {optimal_cores,tail} [--datapoints DATAPOINTS] [--iterations ITERATIONS]
               [--cpus CPUS] [--max-cpus MAX_CPUS] [--interval INTERVAL]
               [--utilization UTILIZATION] [--sla SLA] [--discipline {fifo|sjf}]
               [--service_time [SERVICE_TIME_DISTRIBUTION [Dist args ...]]]
               [--interarrival_time [INTERARRIVAL_TIME_DISTRIBUTION [Dist args ...]]]
               [--quanta PS_QUANTA_US] [--warmup WARMUP_MS] [--dump DUMPFILE]
               [--rps MRPS]

        Supported distributions are:
            constant <val>
            exponential <scale>
            normal <mean> <stddev>
            lognormal <mean> <stddev>
            gpareto <scale> <c>
            uniform <a> <b>
            bimodal <mean1> <percent1> <mean2> <percent2>
            All distribution parameters should be in microseconds

        Supported disciplines:
            fifo
            sjf
            ps
            psjf
            nwc (Non-work-conserving scheduler)
    ----------------------------------------------------------------

    positional arguments:
  {optimal_cores,tail}  Number of rate points to simulate, default=10

optional arguments:
  -h, --help            show this help message and exit
  --datapoints DATAPOINTS
                        Number of rate points to simulate, default=10
  --iterations ITERATIONS
                        Number of simulation iterations, default=1
  --cpus CPUS           Starting number of CPUs to simulate, default=10
  --max-cpus MAX_CPUS   Maximum number of CPUs to simulate, default=16
  --interval INTERVAL   Length of simulation interval in seconds, default=1s
  --utilization UTIL    Number of processor to fully utilize, default=8
  --sla SLA             Service Level Objective for p999 latency, default=200us
  --quanta QUANTA       Processor-sharing/PSJF time quanta in microseconds, default=5us
  --warmup WARMUP       The latency results for this duration will be ignored at the start of the experiments, default=10ms
  --rps RPS             Used in optimal core mode to find the optimal number of core for that given rps, default=0MRPS
  --mean_service_time MEAN_SERVICE_TIME
                        Mean service time is used to automatically determine the max load the system can tolerate, default=10us
  --dump DUMP           You can specify a file for the simulator to dump all the request latencies
  --discipline DISC     Queuing discipline, default=fifo
  --service_time        SERVICE_TIME_DISTRIBUTION [parameters ...]
  --interarrival_time   INTERARRIVAL_TIME_DISTRIBUTION [parameters ...]
```
The only positional argument is `mode`. Select between `optimal_cores,tail`. The `tail` mode reports the latency percentiles for the given rates. The `optimal_cores` mode tries to find the optimal number of servers (cores) for a given rate and SLA combination.

## Example
`$ python3 main.py tail --interarrival_time exponential 200  --service_time exponential 10 --utilization 16 --discipline sjf --max-cpus 16 --datapoints 30 --cpus 16`

Simulates a queuing system with 16 servers, single queue with SJF scheduling the arrivals are exponentially distributed with lambda of 200 microseconds and service times have a lambda of 10microseconds. The simulation will try 30 different rates from near zero to the point where system utilization becomes 1.

The result will be something like this:

```
Util, Requests/s, avg, p50, p90, p95, p99, p999, Cores
0.53, 53333.33, 10.02, 7.00,23.09,29.93,45.78,67.47, 0.0
1.07, 106666.67, 10.00, 6.91,23.03,29.94,46.07,69.44, 0.0
1.60, 160000.00, 9.97, 6.91,22.90,29.83,45.66,69.54, 0.0
2.13, 213333.33, 10.00, 6.94,22.91,29.91,45.77,68.90, 0.0
2.67, 266666.67, 10.00, 6.93,23.06,29.95,46.00,68.99, 0.0
3.20, 320000.00, 9.98, 6.92,22.96,29.88,46.21,69.59, 0.0
3.73, 373333.33, 9.99, 6.92,22.99,29.94,46.05,69.15, 0.0
4.27, 426666.67, 9.99, 6.93,22.99,29.88,45.94,69.12, 0.0
4.80, 480000.00, 9.99, 6.92,23.05,29.94,46.03,68.45, 0.0
5.33, 533333.33, 10.01, 6.94,23.03,29.91,46.05,69.32, 0.0
5.87, 586666.67, 10.00, 6.92,23.06,29.92,46.00,68.59, 0.0
6.40, 640000.00, 10.00, 6.94,23.03,29.89,46.10,68.71, 0.0
6.93, 693333.33, 10.00, 6.94,22.99,29.94,46.12,69.68, 0.0
7.47, 746666.67, 10.00, 6.94,23.00,29.89,46.00,69.13, 0.0
8.00, 800000.00, 10.00, 6.94,23.02,29.92,45.95,69.24, 0.0
8.53, 853333.33, 10.02, 6.95,23.02,29.98,46.12,68.86, 0.0
9.07, 906666.67, 10.02, 6.95,23.04,30.02,46.31,69.35, 0.0
9.60, 960000.00, 10.05, 6.98,23.13,30.09,46.17,69.16, 0.0
10.13, 1013333.33, 10.09, 6.98,23.23,30.27,46.63,69.86, 0.0
10.67, 1066666.67, 10.16, 7.03,23.37,30.41,46.78,69.91, 0.0
11.20, 1120000.00, 10.21, 7.08,23.45,30.52,46.98,71.08, 0.0
11.73, 1173333.33, 10.28, 7.12,23.62,30.81,47.37,71.86, 0.0
12.27, 1226666.67, 10.40, 7.18,23.85,31.19,48.33,73.16, 0.0
12.80, 1280000.00, 10.60, 7.28,24.30,31.86,49.83,78.68, 0.0
13.33, 1333333.33, 10.83, 7.38,24.76,32.59,52.04,84.65, 0.0
13.87, 1386666.67, 11.16, 7.49,25.42,33.74,55.22,98.30, 0.0
14.40, 1440000.00, 11.68, 7.65,26.36,35.49,61.05,125.16, 0.0
14.93, 1493333.33, 12.62, 7.85,27.90,38.54,72.79,190.18, 0.0
15.47, 1546666.67, 14.76, 8.09,30.33,43.83,101.35,437.06, 0.0
16.00, 1600000.00, 197.97, 8.48,35.61,57.21,231.96,12421.20, 0.0
```


## Author
Erfan Sharafzadeh (e.[last name]@jhu.edu)

