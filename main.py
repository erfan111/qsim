from scipy import stats
import numpy as np
import argparse
import heapq
import math
import random
import multiprocessing as mp
from itertools import product
from operator import attrgetter


dists = {
    "exponential": stats.expon,
    "uniform": stats.uniform,
    "normal": stats.norm,
    "gpareto": stats.genpareto
}

class Simulator:
    def __init__(self, interval, n_cpus, max_cpus, iterations, discipline, quanta_us=0):
        self.interval = interval
        self.n_cpus = n_cpus
        self.max_cpus = max_cpus
        self.iterations = iterations
        self.discipline = discipline
        self.quanta_ns = 1000*quanta_us


class Request():
    def __init__(self, start, serv):
        self.start = start
        self.service_time = serv
        self.queue_time = 0
        self.queue_start = 0
        self.runtime = 0
    
    def __repr__(self):
        return "[{}-{}]".format(self.start, self.service_time)


class Distribution:
    def __init__(self, name):
        self.p = {}
        self.dist = name
    
    def sample(self):
        if self.dist == "uniform":
            return dists[self.dist].rvs(loc=self.p["a"], scale=self.p["b"])
        elif self.dist == "exponential":
            a = -math.log(1 - random.random()) / self.p["lambda"]
            return a * 1e6
        elif self.dist == "normal":
            return dists[self.dist].rvs(loc=self.p["mean"], scale=self.p["stddev"])
        elif self.dist == "constant":
            return self.p["value"]
        elif self.dist == "gpareto":
            return dists[self.dist].rvs(c=self.p["c"], scale=self.p["scale"])
        elif self.dist == "lognormal":
            return random.lognormvariate(self.p["mean"], self.p["stddev"])
        elif self.dist == "bimodal":
            if(random.randint(1, 100) < self.p["weight1"]):
                return random.normalvariate(self.p["mean1"], 1)
            else:
                return random.normalvariate(self.p["mean2"], 1)
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
        elif self.dist == "lognormal":
            self.p["mean"] = float(args[1])
            self.p["stddev"] = float(args[2])
        elif self.dist == "bimodal":
            self.p["mean1"] = float(args[1])
            self.p["weight1"] = float(args[2])
            self.p["mean2"] = float(args[3])
            self.p["weight2"] = float(args[4])
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
        elif self.dist == "lognormal":
            self.p["mean"] = l
        else:
            raise NotImplementedError

class Event:
    def __init__(self, event_type, time, req):
        self.type = event_type
        self.time = time
        self.request = req

    def __gt__(self, e2):
        return self.time > e2.time
    
    def __lt__(self, e2):
        return self.time < e2.time

    def __eq__(self, e2):
        return self.time == e2.time
    
    def __ne__(self, e2):
        return self.time != e2.time
    
    def __le__(self, e2):
        return self.time <= e2.time

    def __ge__(self, e2):
        return self.time >= e2.time


def pop_shortest_job(queue):
        sj = min(queue,key=attrgetter('service_time'))
        queue.remove(sj)
        # sj = queue.pop(0)
        return sj

def create_schedule(s_dist, rps, interval, i_dist):
    ns_per_req = 1e9 / rps
    n_reqs = int((interval * 1e9) // ns_per_req) + 1
    i_dist.set_lambda(ns_per_req/1000)
    # print(n_reqs)
    reqs = []
    last = 0
    for _ in range(n_reqs):
        last += (i_dist.sample() *1e3)
        if last >= (interval*1e9):
            break
        last = min([last, interval*1e9])
        ss = s_dist.sample()* 1e3
        reqs.append(Request(last,ss))
    
    return reqs

# Single queue - FIFO
def simulate_schedule_fifo(schedule, interval, n_cpus, max_cpus):
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
    
    # latencies.sort()
    # print(latencies[-10:])
    return latencies

# Single queue - SJF
def simulate_schedule_sjf(schedule, interval, n_cpus, max_cpus):
    time = 0
    idle_cpus = n_cpus
    queue = []
    events = []
    latencies = []
    max_q = 0
    for request in schedule:
        heapq.heappush(events, Event('A', request.start, request))
    while len(events) > 0:
        event = heapq.heappop(events)
        request = event.request
        time = event.time
        if event.type == 'A':
            if idle_cpus > 0:  # a CPU is free
                heapq.heappush(events, Event('D', request.start + request.service_time, request))
                idle_cpus -= 1
            else:
                if(len(queue) > max_q):
                    max_q = len(queue)
                queue.append(event.request)
        else:
            latencies.append(request.service_time + request.queue_time)
            if len(queue) > 0:
                sj = pop_shortest_job(queue)
                sj.queue_time = time - sj.start
                heapq.heappush(events, Event('D', time + sj.service_time, sj))
            else:
                idle_cpus = min(idle_cpus+1, n_cpus)
    
    # latencies.sort()
    # print(latencies[-10:], max_q)
    return latencies


def simulate_schedule_ps(schedule, interval, n_cpus, max_cpus, quanta_ns):
    time = 0
    idle_cpus = n_cpus
    queue = []
    events = []
    latencies = []
    for request in schedule:
        heapq.heappush(events, Event('A', request.start, request))
    while len(events) > 0:
        event = heapq.heappop(events)
        request = event.request
        time = event.time
        if event.type == 'A':
            if idle_cpus > 0:  # a CPU is free
                if request.service_time > quanta_ns:
                    heapq.heappush(events, Event('P', request.start + quanta_ns, request))
                else:
                    heapq.heappush(events, Event('D', request.start + request.service_time, request))
                idle_cpus -= 1
            else:
                event.request.queue_start = time
                queue.append(event.request)
        elif event.type == 'D':
            latencies.append(request.service_time + request.queue_time)
            if len(queue) > 0:
                sj = queue.pop(0)
                sj.queue_time += (time - sj.queue_start)
                if (request.service_time - request.runtime) > quanta_ns:
                    heapq.heappush(events, Event('P', time + quanta_ns, sj))
                else:
                    heapq.heappush(events, Event('D', time + sj.service_time - sj.runtime, sj)) 
            else:
                idle_cpus = min(idle_cpus+1, n_cpus)
        else:   # preemption event
            request.queue_start = time
            queue.append(request)
            request.runtime += quanta_ns
            sj = queue.pop(0)
            sj.queue_time += (time - sj.queue_start)
            if (request.service_time - request.runtime) > quanta_ns:
                heapq.heappush(events, Event('P', time + quanta_ns, sj))
            else:
                heapq.heappush(events, Event('D', time + sj.service_time - sj.runtime, sj))
    return latencies

def simulate_schedule_psjf(schedule, interval, n_cpus, max_cpus, quanta_ns):
    time = 0
    idle_cpus = n_cpus
    queue = []
    events = []
    latencies = []
    for request in schedule:
        heapq.heappush(events, Event('A', request.start, request))
    while len(events) > 0:
        event = heapq.heappop(events)
        request = event.request
        time = event.time
        if event.type == 'A':
            if idle_cpus > 0:  # a CPU is free
                if request.service_time > quanta_ns:
                    heapq.heappush(events, Event('P', request.start + quanta_ns, request))
                else:
                    heapq.heappush(events, Event('D', request.start + request.service_time, request))
                idle_cpus -= 1
            else:
                event.request.queue_start = time
                queue.append(event.request)
        elif event.type == 'D':
            latencies.append(request.service_time + request.queue_time)
            if len(queue) > 0:
                sj = pop_shortest_job(queue)
                sj.queue_time += (time - sj.queue_start)
                if (request.service_time - request.runtime) > quanta_ns:
                    heapq.heappush(events, Event('P', time + quanta_ns, sj))
                else:
                    heapq.heappush(events, Event('D', time + sj.service_time - sj.runtime, sj)) 
            else:
                idle_cpus = min(idle_cpus+1, n_cpus)
        else:   # preemption event
            request.queue_start = time
            queue.append(request)
            request.runtime += quanta_ns
            sj = pop_shortest_job(queue)
            sj.queue_time += (time - sj.queue_start)
            if (request.service_time - request.runtime) > quanta_ns:
                heapq.heappush(events, Event('P', time + quanta_ns, sj))
            else:
                heapq.heappush(events, Event('D', time + sj.service_time - sj.runtime, sj))
    return latencies

def simulate_schedule(schedule, sim):
    if sim.discipline == 'fifo':
        return simulate_schedule_fifo(schedule, sim.interval, sim.n_cpus, sim.max_cpus)
    elif sim.discipline == 'sjf':
        return simulate_schedule_sjf(schedule, sim.interval, sim.n_cpus, sim.max_cpus)
    elif sim.discipline == 'ps':
        return simulate_schedule_ps(schedule, sim.interval, sim.n_cpus, sim.max_cpus, sim.quanta_ns)
    elif sim.discipline == 'psjf':
        return simulate_schedule_psjf(schedule, sim.interval, sim.n_cpus, sim.max_cpus, sim.quanta_ns)
    else:
        raise NotImplementedError("Queue discipline not implemented")

def simulate(s_dist: Distribution, rps, sim, i_dist: Distribution):
    latencies = []
    for _ in range(sim.iterations):
        schedule = create_schedule(s_dist, rps, sim.interval, i_dist)
        latencies += simulate_schedule(schedule, sim)
    tail = np.percentile(latencies, [50,90, 95,99,99.9])/1000
    with open("latencies.csv", "w") as f:
        for l in latencies:
            f.write("{}\n".format(l))
    # print(latencies)
    return ((sum(latencies)/1000)/len(latencies), *tail)

def f(point, args):
    load = (point / args.datapoints) * args.utilization
    rps = (1e6 * load) / 10 #int(args.service_time_distribution[1])
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
    sim = Simulator(args.interval, args.cpus, int(args.max_cpus), args.iterations, args.discipline, quanta_us=args.quanta)
    avg, p50, p90, p95, p99, p999 = simulate(service_dist, rps, sim, iarrival_dist)
    # print("{}, {}, {}, {}".format(load, rps, efficiency, optimal_cpus))  # todo: check the difference between rps vs load
    print("{:.2f}, {:.2f}, {:.2f}, {:.2f},{:.2f},{:.2f},{:.2f},{:.2f}, {}".format(load, rps, avg, p50, p90, p95, p99, p999, optimal_cpus))

# TODO: implement optimal_core mode
def initiate(args):
    print("Load, Requests/s, avg, p50, p90, p95, p99, p999, Cores")
    N = mp.cpu_count()

    with mp.Pool(processes = N) as p:
        points = [(point, args) for point in [28]]# range(1, args.datapoints+1)]
        p.starmap(f, points)
    # for point in range(1, args.datapoints+1):
    #     f(point, args)

    
 


def arguments():
    usg = '''
        main.py [-h] {optimal_cores,tail} [--datapoints DATAPOINTS] [--iterations ITERATIONS]
               [--cpus CPUS] [--max-cpus MAX_CPUS] [--interval INTERVAL]
               [--utilization UTILIZATION] [--sla SLA] [--discipline {fifo|sjf}]
               [--service_time [SERVICE_TIME_DISTRIBUTION [Dist args ...]]]
               [--interarrival_time [INTERARRIVAL_TIME_DISTRIBUTION [Dist args ...]]]
               [--quanta PS_QUANTA_US]

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
               
    '''
    parser = argparse.ArgumentParser(description='Queuing System Simulation Software', usage=usg)
    parser.add_argument('mode', choices=['optimal_cores', 'tail'], default='tail',
                        help='Number of rate points to simulate, default=10')
    parser.add_argument('--datapoints', dest='datapoints', default=10, type=int,
                        help='Number of rate points to simulate, default=10')
    parser.add_argument('--iterations', dest='iterations', default=1, type=int,
                        help='Number of simulation iterations, default=1')
    # parser.add_argument('--rps', dest='max_load', default=500000,
    #                     help='Maximum Request per second')
    parser.add_argument('--cpus', dest='cpus', default=10, type=int,
                        help='Starting number of CPUs to simulate, default=10')
    parser.add_argument('--max-cpus', dest='max_cpus', default=16, type=int,
                        help='Maximum number of CPUs to simulate, default=16')
    parser.add_argument('--interval', dest='interval', default=1, type=int,
                        help='Length of simulation interval in seconds, default=1s')
    parser.add_argument('--utilization', dest='utilization', default=8, type=int,
                        help='Number of processor to fully utilize, default=8')
    parser.add_argument('--sla', dest='sla', default=200, type=int,
                        help='Service Level Objective for p999 latency, default=200us')
    parser.add_argument('--quanta', dest='quanta', default=5, type=float,
                        help='Processor-sharing/PSJF time quanta in microseconds, default=5us')
    parser.add_argument('--discipline', dest='discipline', default='fifo',
                        help='Queuing discipline, default=fifo')
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
