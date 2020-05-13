from scipy import stats
import numpy as np
import argparse
import heapq
import math
import random
import multiprocessing as mp
from itertools import product
from operator import attrgetter

from utils.util import *

MULTIPROCESSING_CPUS = mp.cpu_count()

dists = {
    "exponential": stats.expon,
    "uniform": stats.uniform,
    "normal": stats.norm,
    "gpareto": stats.genpareto
}

class Simulator:
    def __init__(self, interval, n_cpus, max_cpus, iterations, discipline, warmup, quanta_us=0, dump=""):
        self.interval = interval
        self.n_cpus = n_cpus
        self.max_cpus = max_cpus
        self.iterations = iterations
        self.discipline = discipline
        self.warmup = warmup
        self.quanta_ns = 1000*quanta_us
        self.dump = dump


class Request():
    def __init__(self, start, serv, crit=True):
        self.start = start
        self.service_time = serv
        self.queue_time = 0
        self.queue_start = 0
        self.runtime = 0
        self.critical = crit
    
    def __repr__(self):
        return "[{}-{}({})]".format(self.start, self.service_time, self.critical)

class Accounting():
    def __init__(self, warmup, duration):
        self.warmup_ns = warmup*1000
        self.duration_ns = duration*1e9
        self.latencies = []
        self.late_reqs = 0
        self.non_conserved = 0

    
    def add_latency(self, latency, start_time, current_time):
        if start_time > self.warmup_ns:
            # print(start_time, current_time, self.duration_ns)
            if current_time < self.duration_ns:
                self.latencies.append(latency)
            else:
                self.late_reqs += 1

    def get_percentiles_us(self, p):
        return np.percentile(self.latencies, p)/1000

    def dump(self, file):
        with open(file, "w") as f:
            for l in self.latencies:
                f.write("{}\n".format(l))

    def get_mean(self):
        return (sum(self.latencies)/1000)/len(self.latencies)



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

class MQEvent(Event):
    def __init__(self, event_type, time, req, task):
        super().__init__(event_type, time, req)
        self.cpu = 0
        self.task = task


def pop_shortest_job(queue):
        sj = min(queue,key=attrgetter('service_time'))
        queue.remove(sj)
        # sj = queue.pop(0)
        return sj

def create_schedule(s_dist, rps, interval, i_dist):
    # print("creating schedule")
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

def create_mq_schedule(s_dist, rps, interval, i_dist):
    # print("creating schedule")
    LC_TASK_RATIO = 80 # 80% mq tasks
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
        req = Request(last,ss)
        if random.uniform(0,99) > LC_TASK_RATIO:
            req.critical = False
        reqs.append(req)
    
    return reqs

###################################

MQ_PERIOD = 40000
MQ_RUNTIME = 20000

class SchedEntity:
    def __init__(self, request, start_time, service_time, mq_task):
        self.request = request
        self.start_time = start_time
        self.service_time = service_time
        self.queue_time = 0
        self.cpu = 0
        self.status = 1
        self.mq_task = mq_task
        self.sum_exec_runtime = 0
        self.exec_start = 0
        self.queue_start = 0

class Processor:
    def __init__(self, id):
        self.id = id
        self.queue = []
        self.cfsrq = []
        self.nr_running = 0
        self.idle = True
        self.time = 0
        self.quanta_start = 0
        self.curr = None
        self.microq_time = 0
        self.delta_exec_uncharged = 0
        self.delta_exec_total = 0
        self.microq_target_time = 0
        self.microq_time = 0
        self.throttled = False
        self.cfs_nr_running = 0
        self.hr_timer_active = False

class EventManager:
    def __init__(self):
        self.queue = []
        self.time = 0
    
    def add_event(self, event):
        heapq.heappush(self.queue, event)

    def pop_event(self):
        e = heapq.heappop(self.queue)
        self.time = e.time
        return e
    
    def has_event(self):
        return len(self.queue) > 0


class SchedMQ:
    def __init__(self, schedule, interval, n_cpus, max_cpus, accounting, evm):
        self.schedule = schedule
        self.interval = interval
        self.n_cpus = n_cpus
        self.max_cpus = max_cpus
        self.accounting = accounting
        self.period = MQ_PERIOD
        self.runtime = MQ_RUNTIME
        self.curr = None
        self.event_manager = evm
        self.processors = []
        for i in range(n_cpus):
            self.processors.append(Processor(i))

    def account_bandwidth(self, cpu):
        delta = self.processors[cpu].time - self.processors[cpu].quanta_start
        self.processors[cpu].quanta_start += delta
        if not self.processors[cpu].curr:
            return
        if self.processors[cpu].curr.mq_task:
            self.processors[cpu].microq_time += delta
            self.processors[cpu].delta_exec_uncharged += delta
        self.processors[cpu].delta_exec_total += delta
        self.processors[cpu].microq_target_time += (delta * (self.runtime/self.period))
        self.processors[cpu].microq_time = clamp(self.processors[cpu].microq_time, u_saturation_sub(self.processors[cpu].microq_target_time, 2*self.period), self.processors[cpu].microq_target_time + (2*self.period))
    ####

    def enqueue_task(self, task, cpu):
        self.account_bandwidth(cpu)
        task.queue_start = self.processors[task.cpu].time
        self.processors[task.cpu].queue.append(task)
        self.processors[task.cpu].nr_running += 1
        if not self.processors[task.cpu].curr:
            self.resched_curr(cpu)

    def enqueue_cfs_task(self, task, cpu):
        task.queue_start = self.processors[task.cpu].time
        self.processors[task.cpu].cfsrq.append(task)
        self.processors[task.cpu].nr_running += 1
        self.processors[task.cpu].cfs_nr_running += 1
        if not self.processors[task.cpu].curr:
            self.resched_curr(cpu)

    def dequeue_task(self, task, cpu):
        self.update_curr(cpu, task)
        task.queue_time += (self.processors[task.cpu].time - task.queue_start)
        self.processors[cpu].queue.remove(task)
        self.processors[cpu].nr_running -= 1

    def dequeue_cfs_task(self, task, cpu):
        task.queue_time += (self.processors[task.cpu].time - task.queue_start)
        self.processors[cpu].cfsrq.remove(task)
        self.processors[cpu].nr_running -= 1
        self.processors[cpu].cfs_nr_running -= 1

    def requeue_task(self, task, cpu):
        if task.mq_task:
            self.processors[cpu].queue.append(self.processors[cpu].queue.pop(0))
        else:
            self.processors[cpu].cfsrq.append(self.processors[cpu].cfsrq.pop(0))
        task.queue_start = self.processors[task.cpu].time


    def yield_task(self, task, cpu):
        self.requeue_task(task, cpu)

    def timer_needed(self, cpu):
        return self.processors[cpu].nr_running and self.processors[cpu].cfs_nr_running > 0
    
    def check_timer(self, cpu):
        if self.processors[cpu].hr_timer_active:
            return
        self.account_bandwidth(cpu)
        if self.processors[cpu].microq_time < self.processors[cpu].microq_target_time:
		    self.processors[cpu].throttled = False
		    expire = self.processors[cpu].microq_target_time - self.processors[cpu].microq_time
		    expire = max(self.runtime, expire)
        else:
            self.processors[cpu].throttled = True
            expire = self.processors[cpu].microq_time - self.processors[cpu].microq_target_time
            expire = max(expire, self.period - self.runtime)
        
        self.processors[cpu].hr_timer_active = True
        self.event_manager.add_event(MQEvent("T", expire*1000, None, None))

    def pick_next_task(self, task, cpu):
        if self.processors[cpu].nr_running == 0:
            return None
        if self.timer_needed(cpu):
            self.check_timer(cpu)
            if self.processors[cpu].throttled:
                return None
        else:
            self.processors[cpu].throttled = False

        self.put_prev_task(task, cpu)
        next_task = self.processors[cpu].queue[0]
        return next_task

    def put_prev_task(self, task, cpu):
        pass
        # if self.processors[cpu].delta_exec_uncharged * self.period > self.processors[cpu].delta_exec_total * self.runtime:
        #      contrib_ratio = runtime * (1 << 10)/ period
        # else:
        #     contrib_ratio = (1 << 10)
        # update rt load avg ratio

    def find_rq(self, task, cpu):
        low_prio_cpu = -1
        low_nmicroq = self.processors[task.cpu].nr_running - self.processors[task.cpu].cfs_nr_running
        low_nmicroq_cpu = -1
        for c in range(task.cpu, self.n_cpus):
            if self.processors[c].nr_running == 0:
                return c
        for c in range(0, task.cpu):
            if self.processors[c].nr_running == 0:
                return c
        for c in range(0, self.n_cpus):
            if self.processors[c].nr_running == self.processors[c].cfs_nr_running:
                low_prio_cpu = c
            elif (self.processors[c].nr_running - self.processors[c].cfs_nr_running + 1 < low_nmicroq):
                low_nmicroq_cpu = c
                low_nmicroq = self.processors[c].nr_running - self.processors[c].cfs_nr_running
        if low_prio_cpu != -1:
            return low_prio_cpu
        return low_nmicroq_cpu

    def find_cfs_rq(self, task, cpu):
        low_ncfs = self.processors[task.cpu].cfs_nr_running
        low_ncfs_cpu = task.cpu
        for c in range(0, self.n_cpus):
            if self.processors[c].nr_running == 0:
                return c
        for c in range(0, self.n_cpus):
            if self.processors[c].cfs_nr_running < low_ncfs:
                low_ncfs_cpu = c
                low_ncfs = self.processors[c].nr_running - self.processors[c].cfs_nr_running
        return low_ncfs_cpu

    def select_task_rq(self, task, cpu):
        if self.processors[cpu].nr_running > 0:
            target = self.find_rq(task, cpu)
            if target != -1:
                task.cpu = target

    def select_cfs_task_rq(self, task, cpu):
        if self.processors[cpu].nr_running > 0:
            target = self.find_cfs_rq(task, cpu)
            if target != -1:
                task.cpu = target

    def push_microq_task(self, cpu):
        return 0

    def task_woken(self, task, cpu):
        if self.processors[cpu].nr_running - self.processors[cpu].cfs_nr_running > 0:
            while self.push_microq_task(cpu):
                pass

    def task_tick(self):
        pass

    def resched_curr(self, cpu):
        current_task = self.processors[cpu].curr
        if current_task:
            if current_task.sum_exec_runtime - current_task.service_time <= 0:  # task is finished!
                current_task.queue_start = self.event_manager.time
                current_task.status = 0 # Finished
                if current_task.mq_task:
                    self.dequeue_task(current_task, cpu)
                else:
                    self.dequeue_cfs_task(current_task, cpu)
                self.processors[cpu].curr = None
            else:
                self.requeue_task(current_task, cpu)
        next_task = self.pick_next_task(current_task, cpu)
        if next_task:
            # print("context switching")
            self.processors[cpu].curr = next_task
            next_task.queue_time += (self.processors[next_task.cpu].time - next_task.queue_start)
            self.event_manager.add_event(MQEvent("D", self.event_manager.time + next_task.service_time, None, next_task))
        else:
            cfs_task = self.pick_next_task_cfs(current_task, cpu)
            if cfs_task:
            # print("context switching")
                self.processors[cpu].curr = cfs_task
                next_task.queue_time += (self.processors[cfs_task.cpu].time - next_task.queue_start)
                self.event_manager.add_event(MQEvent("D", self.event_manager.time + cfs_task.service_time, None, cfs_task))

    def mq_period_timer(self, cpu, task):
        nextslice = self.period
        self.processors[cpu].hr_timer_active = False
        self.account_bandwidth(cpu)
        if self.timer_needed(cpu):
            if self.processors[cpu].throttled:
               self.processors[cpu].throttled = False
               ns = u_saturation_sub(self.processors[cpu].microq_target_time, self.processors[cpu].microq_time)
               nextslice = max(ns, self.runtime)
            else:
                self.processors[cpu].throttled = True
                ns = u_saturation_sub(self.processors[cpu].microq_time, self.processors[cpu].microq_target_time)
                ns = ns*self.period/self.runtime
                ns = clamp(ns, u_saturation_sub(self.period, self.runtime), u_saturation_sub(self.period, self.runtime/2))
            self.resched_curr(cpu)
            self.processors[cpu].hr_timer_active = True
            self.event_manager.add_event(MQEvent("T", self.event_manager.time + nextslice, None, task))
        else:
            self.processors[cpu].throttled = False

    def update_curr(self, cpu, curr):
        if not curr.mq_task:
            return
        self.account_bandwidth(cpu)
        curr.sum_exec_runtime += self.processors[cpu].delta_exec_uncharged
        curr.exec_start = self.processors[cpu].time
         # update rt load avg ratio
        self.processors[cpu].delta_exec_uncharged = 0
        self.processors[cpu].delta_exec_total = 0




def simulate_sched_microquanta(schedule, interval, n_cpus, max_cpus, accounting):
    time = 0
    events = EventManager()
    mq = SchedMQ(schedule, interval, n_cpus, max_cpus, accounting, events)
    for request in schedule:
        t = SchedEntity(request, request.start, request.service_time, request.critical)
        e = MQEvent('A', request.start, request, t)
        # mq.select_task_rq(t, 0)
        events.add_event(e)
    while events.has_event():
        event = events.pop_event()
        request = event.request
        time = event.time
        task = event.task
        mq.processors[task.cpu].time = time
        if event.type == 'A':
            if event.task.mq_task:
                mq.select_task_rq(event.task, 0)
                mq.enqueue_task(task, task.cpu)
            else:
                mq.select_cfs_task_rq(event.task, 0)
                mq.enqueue_cfs_task(task, task.cpu)
        elif event.type == 'D':
            if event.task.status == 1:
                mq.resched_curr(task.cpu)
                # print(task.sum_exec_runtime, task.queue_time)
                if task.mq_task:
                    accounting.add_latency(task.service_time + task.queue_time, task.start_time, time)
        elif event.type == 'T':
            mq.mq_period_timer(task.cpu, task)
        else:
            print("WTF!")


###################################

# Single queue - FIFO
def simulate_schedule_fifo(schedule, interval, n_cpus, max_cpus, accounting):
    time = 0
    cpus = [0] * n_cpus
    for _ in range(n_cpus, max_cpus):
        cpus.append(1e6 * interval)
    heapq.heapify(cpus)
    for request in schedule:
        time = max(time, request.start)
        time = max(time, heapq.heappop(cpus))
        accounting.add_latency(time - request.start + request.service_time, request.start, time)
        # latencies.append(time - request.start + request.service_time)
        heapq.heappush(cpus, request.service_time + time)

# Single queue - SJF
def simulate_schedule_sjf(schedule, interval, n_cpus, max_cpus, accounting):
    # print("simulating schedule")
    time = 0
    idle_cpus = n_cpus
    queue = []
    events = []
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
            accounting.add_latency(request.service_time + request.queue_time, request.start, time)
            if len(queue) > 0:
                sj = pop_shortest_job(queue)
                sj.queue_time = time - sj.start
                heapq.heappush(events, Event('D', time + sj.service_time, sj))
            else:
                idle_cpus = min(idle_cpus+1, n_cpus)
    


def simulate_schedule_ps(schedule, interval, n_cpus, max_cpus, quanta_ns, accounting):
    time = 0
    idle_cpus = n_cpus
    queue = []
    events = []
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
            accounting.add_latency(request.service_time + request.queue_time, request.start, time)
            # latencies.append(request.service_time + request.queue_time)
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


def simulate_schedule_nwc(schedule, interval, n_cpus, max_cpus, quanta_ns, accounting):
    # print("simulating schedule")
    THRESHOLD = 4000
    time = 0
    idle_cpus = n_cpus
    queue = []
    events = []
    max_q = 0
    e_cnt = 0
    for request in schedule:
        heapq.heappush(events, Event('A', request.start, request))
    while len(events) > 0:
        e_cnt += 1
        # if e_cnt%10000 == 0:
        #     print("simulating event {} of {}".format(e_cnt, len(schedule)))
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
            accounting.add_latency(request.service_time + request.queue_time, request.start, time)
            if len(events) < 1:
                return
            if (events[0].type == 'A' and events[0].time - time < THRESHOLD) or len(queue) < 1:
                if len(queue) > 0:
                    accounting.non_conserved +=1
                    # print("work not conserved! {} q = {}".format(work_not_conserved, len(queue)))
                # print(events[0].time - time, len(queue))
                idle_cpus = min(idle_cpus+1, n_cpus)
            else:
                sj = pop_shortest_job(queue)
                sj.queue_time = time - sj.start
                heapq.heappush(events, Event('D', time + sj.service_time, sj))
                

def simulate_schedule_psjf(schedule, interval, n_cpus, max_cpus, quanta_ns, accounting):
    time = 0
    idle_cpus = n_cpus
    queue = []
    events = []
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
            accounting.add_latency(request.service_time + request.queue_time, request.start, time)
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

def simulate_schedule(schedule, sim, accounting):
    if sim.discipline == 'fifo':
        simulate_schedule_fifo(schedule, sim.interval, sim.n_cpus, sim.max_cpus, accounting)
    elif sim.discipline == 'sjf':
        simulate_schedule_sjf(schedule, sim.interval, sim.n_cpus, sim.max_cpus, accounting)
    elif sim.discipline == 'ps':
        simulate_schedule_ps(schedule, sim.interval, sim.n_cpus, sim.max_cpus, sim.quanta_ns, accounting)
    elif sim.discipline == 'psjf':
        simulate_schedule_psjf(schedule, sim.interval, sim.n_cpus, sim.max_cpus, sim.quanta_ns, accounting)
    elif sim.discipline == 'nwc':
        simulate_schedule_nwc(schedule, sim.interval, sim.n_cpus, sim.max_cpus, sim.quanta_ns, accounting)
    else:
        raise NotImplementedError("Queue discipline not implemented")

def simulate(s_dist: Distribution, rps, sim, i_dist: Distribution):
    accounting = Accounting(sim.warmup, sim.interval)
    for _ in range(sim.iterations):
        if sim.discipline == 'mq':
            schedule = create_mq_schedule(s_dist, rps, sim.interval, i_dist)
            simulate_sched_microquanta(schedule, sim.interval, sim.n_cpus, sim.max_cpus, accounting)
        else:
            schedule = create_schedule(s_dist, rps, sim.interval, i_dist)
            simulate_schedule(schedule, sim, accounting)
    # print("calculating percentiles")
    tail = accounting.get_percentiles_us([50,90, 95,99,99.9])
    if(sim.dump != ""):
        accounting.dump(sim.dump)
    return (accounting.non_conserved, accounting.late_reqs, accounting.get_mean(), *tail)

def start_optimal_core_mode(args):
    optimal_cpus = 0.0
    efficiency = 0.0
    service_dist = Distribution(args.service_time_distribution[0])
    service_dist.parse_args(args.service_time_distribution)
    iarrival_dist = Distribution(args.interarrival_time_distribution[0])
    iarrival_dist.parse_args(args.interarrival_time_distribution)
    for n_cpus in range(1, args.cpus+1):
        sim = Simulator(args.interval, n_cpus, int(args.max_cpus), args.iterations, args.discipline, int(args.warmup), quanta_us=args.quanta, dump=args.dump)
        late, avg, p50, p90, p95, p99, p999 = simulate(service_dist, args.rps, sim, iarrival_dist)
        print("p999 = {}, cpus = {}, rps = {}".format(p999, n_cpus, args.rps*1e6))
        if p999 < args.sla:
            optimal_cpus = n_cpus
            efficiency = args.rps / optimal_cpus
            break
    print("{}, {}, {}".format(args.rps, efficiency, optimal_cpus))


def f(point, args):
    load = 0
    rps = 0
    if point == 0:
        load = 1
        rps = args.rps*1e6
    else:
        load = (point / args.datapoints) * args.utilization
        rps = (1e6 * load) / args.mean_service_time
    service_dist = Distribution(args.service_time_distribution[0])
    service_dist.parse_args(args.service_time_distribution)
    iarrival_dist = Distribution(args.interarrival_time_distribution[0])
    iarrival_dist.parse_args(args.interarrival_time_distribution)
    sim = Simulator(args.interval, args.cpus, int(args.max_cpus), args.iterations, args.discipline, int(args.warmup), quanta_us=args.quanta, dump=args.dump)
    non_cnsrvd, late, avg, p50, p90, p95, p99, p999 = simulate(service_dist, rps, sim, iarrival_dist)
    print("{:.2f}, {:.2f}, {:.2f}, {:.2f},{:.2f},{:.2f},{:.2f},{:.2f}, {}, {}".format(load, rps, avg, p50, p90, p95, p99, p999, late, non_cnsrvd))

def initiate(args):
    if args.mode == "optimal_core":
        print("Load, Requests/s, avg, p50, p90, p95, p99, p999, Cores")
        start_optimal_core_mode(args)
    else:
        if args.rps != 0:
            f(0, args)
        else:
            with mp.Pool(processes = MULTIPROCESSING_CPUS) as p:
                points = [(point, args) for point in range(1, args.datapoints+1)]
                p.starmap(f, points)
        # for point in range(1, args.datapoints+1):
        #     f(point, args)

    
 


def arguments():
    usg = '''
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
    parser.add_argument('--warmup', dest='warmup', default=10, type=float,
                        help='The latency results for this duration will be ignored at the start of the experiments, default=10ms')
    parser.add_argument('--rps', dest='rps', default=0.0, type=float,
                        help='Used in optimal core mode to find the optimal number of core for that given rps, default=0MRPS')
    parser.add_argument('--mean_service_time', dest='mean_service_time', default=10, type=float,
                        help='Mean service time is used to automatically determine the max load the system can tolerate, default=10us')
    parser.add_argument('--dump', dest='dump', default="", type=str,
                        help='You can specify a file for the simulator to dump all the request latencies')
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
