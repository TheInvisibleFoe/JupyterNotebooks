import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from Likelihood import Logl
def plotll(x, theta, true):
    
    # param = genparam
    # log likelihood array initialization
    N = 1000
    
    logPROBD = np.zeros(N)
    logPROBA = np.zeros(N)
    logPROBF = np.zeros(N)
    logPROBMN = np.zeros(N)

    alpha_range = np.linspace(0, 2, N)
    d_range=np.linspace(0.001,2,N)
    F_L_range=np.linspace(-1,1,N)
    MN_range = np.linspace(0, 50, N)

    t_clean_free = np.array(x[0])
    xmeasured = np.array(x[1])
    xtrue = np.array(x[2])

    for i in range(0,N):
        # log likelihood for the applied force over a range of values    
        D = d_range[i]
        A = alpha_range[i]
        F = F_L_range[i]
        M = MN_range[i]
        logPROBD[i] = Logl(xtrue,[D,theta[1],theta[2],theta[3]])
        logPROBA[i] = Logl(xtrue,[theta[0],A,theta[2],theta[3]])
        logPROBF[i] = Logl(xtrue,[theta[0],theta[1],F,theta[3]])
        logPROBMN[i]= Logl(xtrue,[theta[0],theta[1],theta[2],M])
    
    fig = plt.figure(figsize = (15,9))

    # Gridspec shit have to read later, boiler plate code from matplotlib docs
    gs0 = gridspec.GridSpec(1, 2, figure=fig)

    gs00 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs0[0])

    ax1 = fig.add_subplot(gs00[:, :])

    gs01 = gs0[1].subgridspec(2, 2)
    ax0 = fig.add_subplot(gs00[:, :])
    gs00 = gs0[1].subgridspec(2, 2)
    ax1 = fig.add_subplot(gs01[:1, :1])
    ax3 = fig.add_subplot(gs01[1, :1])
    ax2 = fig.add_subplot(gs01[:1, -1])
    ax4 = fig.add_subplot(gs01[-1,1 ])

    print(logPROBD)

    # convert the loglikelihood to likelihood and normalize them
    # the variable name might be confusing, ll is likelihood 
    llA = np.exp(logPROBA - max(logPROBA))
    llD = np.exp(logPROBD - max(logPROBD))
    llF = np.exp(logPROBF - max(logPROBF))
    llMN = np.exp(logPROBMN - max(logPROBMN))
    # mllA = d_range[np.where(llA == 0)[0][0]]

    plt.suptitle("The Likelihood of functions")

    ax0.plot(t_clean_free, xtrue,'.-b',label = r"Actual Position")
    ax0.plot(t_clean_free, xmeasured,'o--g',label = r"Measured Positions")
    ax0.grid()
    ax0.legend()

    ax1.plot(d_range, llD, label = r"$D_0$")
    ax1.axvline(x = true[0],ls ='--', color = 'r', label = r"$D_0^{true}$")
    ax1.set_title(r"$D_0$")
    ax1.set_xlim((0,2))
    ax1.grid()
    ax1.legend()

    # axes[-1][0].plot(d_range, np.exp(logPROBD1 - max(logPROBD1)), label = r"$\alpha$")

    ax2.plot(alpha_range, llA, label = r"$\alpha$")
    ax2.axvline(x = true[1],ls ='--', color = 'r', label = r"$\alpha^{true}$")
    ax2.set_title(r"$\alpha$")
    ax2.grid()
    ax2.legend(loc = "right")

    ax3.plot(F_L_range, llF, label = r"$f$")
    ax3.axvline(x = true[2],ls ='--', color = 'r', label = "True Value")
    ax3.set_title(r"$f$")
    ax3.grid()
    ax3.legend()

    ax4.plot(MN_range, llMN, label = r"$\sigma^2_{mn}$")
    ax4.axvline(x = true[3],ls ='--', color = 'r', label = "True Value")
    ax4.set_title(r"$f$")
    ax4.grid()
    ax4.legend()

    plt.show()