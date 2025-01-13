import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import scipy.stats as stats
import scipy.special as spec
from numba_stats import expon

# from numba_stats import norm

# @jit
def ModelGen2(x_0, v_0, gamma, t_e, MU, A):
    timesum = 0
    exp = stats.expon
    flag = True
    # tt = [0.]
    posn =[]
    temposn = [x_0]
    fl = 1
    nDT = 100
    while flag == True:
       t =  exp.rvs(scale = 1/gamma)
       s = timesum + t
       if s < t_e:
           timesum += t
           DT = t/nDT
           for i in range(0, nDT-1):
               force = -MU * np.sign(temposn[i]) * (np.abs(temposn[i])**(A-1))
               temposn.append(temposn[i] +  force * DT + v_0 * fl * DT)
           fl*=-1
           posn.extend(temposn)
           endpt = temposn[-1]
           temposn = [endpt]
       elif s == t_e:
           timesum += t
           DT = t/nDT
           for i in range(0, nDT-1):
               force = -MU * np.sign(temposn[i]) * (np.abs(temposn[i])**(A-1))
               temposn.append(temposn[i] +  force * DT + v_0 * fl * DT)
           fl*=-1
           posn.extend(temposn)
           flag = False
       else:
           endt = t_e - timesum
           timesum += endt
           DT = endt/nDT
           for i in range(0, nDT-1):
               force = -MU * np.sign(temposn[i]) * (np.abs(temposn[i])**(A-1))
               temposn.append(temposn[i] +  force * DT + v_0 * fl * DT)
           fl*=-1
           posn.extend(temposn)
           flag = False
    return posn

def prob(x, g, mu, v_0):
    fr = g/mu
    phi = fr -1
    bet = spec.beta(fr, fr)
    p = 2/(4**fr * bet) * (mu/v_0) *(1 - (mu*x/v_0)**2)**phi
    return p
    
x_0 = -4
v_0 = 1
A = 2
mu = 1
gamma = 0.5
t_e = 50
# nsteps = 5
NUM =10
fig,axes = plt.subplots(1,2, figsize = (16,9))
step = np.zeros(NUM)
for i in range(0,NUM):    
    c = ModelGen2(x_0, v_0, gamma, t_e, mu, A)
    posnobsfl = np.array(c)
    nsteps = len(posnobsfl)
    time = np.linspace(0,t_e, nsteps)
    step[i] = posnobsfl[-1]
    axes[0].plot(time, posnobsfl)


n,bins, patches = axes[1].hist(step, bins = 100, density = True, color = 'b');

# Now we find the center of each bin from the bin edges
# bins_mean = [0.5 * (bins[i] + bins[i+1]) for i in range(len(n))]
# axes[1] = fig.add_subplot(122)
# axes[1].scatter(bins_mean, n)

lll = np.linspace(-1,1, 1000)
lly = prob(lll, gamma, mu, v_0)
axes[1].plot(lll,lly, color ='red')
axes[0].axhline(v_0/mu, color = 'black')
axes[0].axhline(-v_0/mu, color = 'black')
axes[0].set_xlim(right = t_e)
# fig.savefig("RunandTumble.png")
plt.show()