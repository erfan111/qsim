## usage: python3 plot.py on.csv off.csv

import numpy as np
import sys
import matplotlib.pyplot as plt
# from scipy.interpolate import spline

font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 14}
font_label = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 13}

plt.rc('font', **font)
plt.rc('legend', fontsize=11)

####################

fifo = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=1)
psjf = np.genfromtxt(sys.argv[2], delimiter=',', skip_header=1)

fig, ax = plt.subplots()

plt.xlabel('Latency ' + r'$\mu$' + "s")
plt.ylabel('CDF')

x = np.arange(len(fifo))

# x_shenango_performance = np.arange(0, 3, 3/len(shenango_performance))
# x_shenango_noidle = np.arange(0, 3, 3/len(shenango_noidle))

# xnew = np.linspace(x.min(),x.max(),300)

# y1_smooth = spline(x,idle[:,5],xnew)
# y2_smooth = spline(x,noidle[:,5],xnew)

# y3_smooth = spline(x,idle[:,7],xnew)
# y4_smooth = spline(x,noidle[:,7],xnew)

# y5_smooth = spline(x,idle[:,8],xnew)
# y6_smooth = spline(x,noidle[:,8],xnew)

# y7_smooth = spline(x,idle[:,9],xnew)
# y8_smooth = spline(x,noidle[:,9],xnew)
# shenango_performance_smooth = spline(x_shenango_performance,shenango_performance,xnew_shenango_performance)
# shenango_noidle_smooth = spline(x_shenango_noidle,shenango_noidle,xnew_shenango_noidle)


# plt.xticks(ind, ('20%', '30%', '40%', '50%'))

plt.ylim(0.9, 1.01)
# plt.ylim(80, 165)
plt.xlim(-1,2000)
fifo = fifo/1000
psjf = psjf/1000

plt.grid(axis="y")
fs = np.sort(fifo)
p = 1. * np.arange(len(fifo))/(len(fifo) - 1)
pp1 = plt.plot(fs, p,)

ps = np.sort(psjf)
p2 = 1. * np.arange(len(psjf))/(len(psjf) - 1)
pp1, pp2 = plt.plot(fs,p, ps, p2)

# plt.xticks(np.arange(0, 3, 0.1))
# ax.set_xticklabels(np.arange(0, 3, 0.1))

plt.legend((pp1, pp2),('FIFO', 'PSJF'))
plt.savefig('sim.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

