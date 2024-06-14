import numpy as np
def Model(theta, genparam):
    """This generates the model for the given parameters

    Args:
        theta (list): lists the parameters of the model in the order 0: D_0, 1: alpha, 2: f, 3: measurement noise
        genparam (list): these are preliminary parameters for the Model 0:DT, 1:t_0, 2:t_e, 3:x_0, 4:k_BT
    """
    
    d = theta[0]
    a = theta[1]
    f = theta[2]
    vmn = theta[3]
    
    DT = genparam[0]
    t_0 = genparam[1]
    t_e = genparam[2]
    x_0 = genparam[3]
    k_BT = genparam[4]
    
    
    steps = int((t_e - t_0)/DT)
    
    p = np.zeros(steps)
    pm = np.zeros(steps)
    t = np.zeros(steps)
    
    p[0] = x_0
    pm[0] = p[0]
    t[0] = t_0
    
    mobil = d * (p[0]**a)
    diffmobil = a * (p[0] ** (a - 1)) * np.sign(p[0])
    
    for i in range(1, steps):
        p[i] = p[i-1] + mobil * f * DT + (k_BT * diffmobil) *DT + np.sqrt(2 * k_BT * mobil) * np.random.normal(0,1) * DT
        t[i] = t[i-1] + DT
        pm[i] = p[i] + np.random.normal(0, np.sqrt(vmn))
        mobil = d * (p[i-1]**a)
        diffmobil = a * (p[i-1] ** (a - 1)) * np.sign(p[i-1])   
        
    return t, p, pm
    