import numpy as np
from numba import njit
@njit
def Logl(x, theta):
    """_summary_

    Args:
        x (list): _description_
        theta (list): _description_
        genparam (list)): _description_
    """
    
    d = theta[0]
    a = theta[1]
    F = theta[2]
    vmn = theta[3]
    
    DT = 1
    # t_0 = genparam[1]
    # t_e = genparam[2]
    # x_0 = genparam[3]
    k_BT = 1
    
    n = len(x)

    #init value
    logl = 0
    

    # Intial values of the variance mean both noisy and clean
    mob = d * (np.abs(x[0]) ** a)
    std_dev_noise = 2 * mob * DT + 2 * vmn
    noisy_mean = x[0] + (mob * F + k_BT * (a * d * ((np.abs(x[0]) ** (a - 1)) * (np.sign(x[0]))))) * DT
    
    # log likelihood at 1st step
    logl = logl - ((x[1] - noisy_mean) ** 2) / (2 * std_dev_noise) - np.log(np.sqrt(2 * np.pi * std_dev_noise))
    
    for i in range(2,n):
        # Mobility at the step
        mob = d * np.abs(x[i - 1]) ** a

        # clean variance at the jth step
        std_dev = 2 * mob * DT

        # clean mean position at the jth time step
        mean_dist = x[i - 1] + (mob * F + a * d * (np.abs(x[i - 1]) ** (a - 1)) * (np.sign(x[i - 1]))) * DT

        # noisy mean position at the jth step
        noisy_mean = mean_dist - vmn/(std_dev_noise) * (x[i-1] - noisy_mean)

        # noisy variance at the jth x
        std_dev_noise = std_dev + vmn * (2 - vmn / std_dev_noise)

        # calculation of log likelihood
        logl = logl - ((x[i] - noisy_mean) ** 2) / (2 * std_dev_noise) - 0.5 * np.log(2 * np.pi * std_dev_noise)
    
    return logl