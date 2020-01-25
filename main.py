from scipy import stats
import numpy as np
import argparse
import heapq
import math
import random

dists = {
    "exponential": stats.expon,
    "uniform": stats.uniform,
    "normal": stats.norm,
    "gpareto": stats.genpareto
}

class Request():
    def __init__(self, start, serv):
        self.start = start
        self.service_time = serv


class Distribution:
    def __init__(self, name):
        self.p = {}
        self.dist = name
    
    def sample(self):
        if self.dist == "uniform":
            return dists[self.dist].rvs(loc=self.p["a"], scale=self.p["b"])
        elif self.dist == "exponential":
            # return dists[self.dist].rvs((self.p["lambda"]))
            a = -math.log(1 - random.random()) / self.p["lambda"]
            # print(a)
            return a * 1e6
        elif self.dist == "normal":
            return dists[self.dist].rvs(loc=self.p["mean"], scale=self.p["stddev"])
        elif self.dist == "constant":
            return self.p["value"]
        elif self.dist == "gpareto":
            return dists[self.dist].rvs(c=self.p["c"], scale=self.p["scale"])
        else:
            raise NotImplementedError
    
    def parse_args(self, args :list):
        if self.dist == "exponential":
            self.p["lambda"] = 1e6/ float(args[1])
        elif self.dist == "uniform":
            self.p["a"] = float(args[1])
            self.p["b"] = float(args[2])
        elif self.dist == "normal":
            self.p["mean"] = float(args[1])
            self.p["stddev"] = float(args[2])
        elif self.dist == "constant":
            self.p["value"] = float(args[1])
        elif self.dist == "gpareto":
            self.p["c"] = float(args[2])
            self.p["scale"] = float(args[1])
        else:
            raise NotImplementedError

    def set_lambda(self, l):
        if self.dist == "exponential":
            self.p["lambda"] = 1e6/l
        elif self.dist == "uniform":
            self.p["a"] += l
            self.p["b"] = l - self.p[2]
        elif self.dist == "normal":
            self.p["mean"] = l
        elif self.dist == "constant":
            self.p["value"] = l
        elif self.dist == "gpareto":
            self.p["scale"] = l
        else:
            raise NotImplementedError

def create_schedule(s_dist, rps, interval, i_dist):
    ns_per_req = 1e9 / rps
    n_reqs = int((interval * 1e9) // ns_per_req) + 1
    i_dist.set_lambda(ns_per_req/1000)
    # print(n_reqs)
    reqs = []
    last = 0
    for _ in range(n_reqs):
        last += (i_dist.sample() *1e3)
        last = min([last, interval*1e9])
        ss = s_dist.sample()* 1e3
        reqs.append(Request(last,ss))
    
    return reqs


def simulate_schedule(schedule, interval, n_cpus, max_cpus):
    time = 0
    cpus = [0] * n_cpus
    for _ in range(n_cpus, max_cpus):
        cpus.append(1e6 * interval)
    heapq.heapify(cpus)
    latencies = []
    for request in schedule:
        time = max(time, request.start)
        time = max(time, heapq.heappop(cpus))
        latencies.append(time - request.start + request.service_time)
        heapq.heappush(cpus, request.service_time + time)
    
    return latencies



def simulate(s_dist: Distribution, rps, interval, n_cpus, max_cpus, iterations, i_dist: Distribution):
    latencies = []
    for _ in range(iterations):
        schedule = create_schedule(s_dist, rps, interval, i_dist)
        latencies += simulate_schedule(schedule, interval, n_cpus, max_cpus)
    tail = np.percentile(latencies, [50,90, 95,99,99.9])/1000
    return ((sum(latencies)/1000)/len(latencies), *tail)


def initiate(args):
    print("Load, Requests/s, avg, p50, p90, p95, p99, p999, Cores")
    for point in range(1, args.datapoints+1):
        load = (point / args.datapoints) * args.utilization
        rps = (1e6 * load) / int(args.service_time_distribution[1])
        latencies = []
        optimal_cpus = 0.0
        efficiency = 0.0
        service_dist = Distribution(args.service_time_distribution[0])
        service_dist.parse_args(args.service_time_distribution)
    
        iarrival_dist = Distribution(args.interarrival_time_distribution[0])
        iarrival_dist.parse_args(args.interarrival_time_distribution)
        # for n_cpus in range(1, args.cpus+1):
        #     tail = simulate(service_dist, rps, args.interval, n_cpus, args.max_cpus, args.iterations, iarrival_dist)
        #     latencies.append(tail)
        #     print("Tail = {}, cpus = {}, rps = {}".format(tail, n_cpus, rps))
        #     if tail < args.sla:
        #         optimal_cpus = n_cpus
        #         efficiency = rps / optimal_cpus
        #         break
        avg, p50, p90, p95, p99, p999 = simulate(service_dist, rps, args.interval, args.cpus, args.max_cpus, args.iterations, iarrival_dist)
        # print("{}, {}, {}, {}".format(load, rps, efficiency, optimal_cpus))  # todo: check the difference between rps vs load
        print("{:.2f}, {:.2f}, {:.2f}, {:.2f},{:.2f},{:.2f},{:.2f},{:.2f}, {}".format(load, rps, avg, p50, p90, p95, p99, p999, optimal_cpus))  # todo: check the difference between rps vs load


def arguments():
    usg = '''
        main.py [-h] {optimal_cores,tail} [--datapoints DATAPOINTS] [--iterations ITERATIONS]
               [--cpus CPUS] [--max-cpus MAX_CPUS] [--interval INTERVAL]
               [--utilization UTILIZATION] [--sla SLA]
               [--service_time [SERVICE_TIME_DISTRIBUTION [Dist args ...]]]
               [--interarrival_time [INTERARRIVAL_TIME_DISTRIBUTION [Dist args ...]]]

        Supported distributions are:
            constant <val>
            exponential <scale>
            normal <mean> <stddev>
            gpareto <scale> <c>
            uniform <a> <b>
            All distribution parameters should be in microseconds
               
    '''
    parser = argparse.ArgumentParser(description='Queuing System Simulation Software', usage=usg)
    parser.add_argument('mode', choices=['optimal_cores', 'tail'], default='tail',
                        help='Number of rate points to simulate, default=10')
    parser.add_argument('--datapoints', dest='datapoints', default=10,
                        help='Number of rate points to simulate, default=10')
    parser.add_argument('--iterations', dest='iterations', default=1,
                        help='Number of simulation iterations, default=1')
    # parser.add_argument('--rps', dest='max_load', default=500000,
    #                     help='Maximum Request per second')
    parser.add_argument('--cpus', dest='cpus', default=10,
                        help='Starting number of CPUs to simulate, default=10')
    parser.add_argument('--max-cpus', dest='max_cpus', default=16,
                        help='Maximum number of CPUs to simulate, default=16')
    parser.add_argument('--interval', dest='interval', default=1,
                        help='Length of simulation interval in seconds, default=1s')
    parser.add_argument('--utilization', dest='utilization', default=8, type=int,
                        help='Number of processor to fully utilize, default=8')
    parser.add_argument('--sla', dest='sla', default=200,
                        help='Service Level Objective for p999 latency, default=200us')
    parser.add_argument('--service_time', dest='service_time_distribution', nargs="*", default=["constant", 1],
                        help='Service-time distribution')
    parser.add_argument('--interarrival_time', dest='interarrival_time_distribution', nargs="*", default=["exponential", 10],
                        help='Interarrival-time distribution')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    parsed_arguments = arguments()
    print(parsed_arguments)
    initiate(parsed_arguments)
