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
sjf = np.genfromtxt(sys.argv[2], delimiter=',', skip_header=1)

fig, ax = plt.subplots()

plt.ylabel('Latency ' + r'$\mu$' + "s")
plt.xlabel('Incoming request rate (MRPS)')

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

plt.ylim(0, 1000)
# plt.ylim(80, 165)
# plt.xlim(0.15,3)
plt.grid(axis="y")
p1, p10 = plt.plot(fifo[:,1]/1e6, fifo[:,2], 'r', sjf[:,1]/1e6, sjf[:,2], 'g', linewidth=3)
p2, p20 = plt.plot(fifo[:,1]/1e6, fifo[:,7], 'r:', sjf[:,1]/1e6, sjf[:,7], 'g:', linewidth=3)

# plt.xticks(np.arange(0, 3, 0.1))
# ax.set_xticklabels(np.arange(0, 3, 0.1))

plt.legend((p1, p2, p10, p20), ('FIFO-mean', 'FIFO-99.9', "SJF-mean", 'SJF-99.9'))
plt.savefig('sim.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

