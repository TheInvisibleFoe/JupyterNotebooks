import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from Likelihood import Logl
from plotLikelihood import plotll

def MODELPROB(Z):
    mp = np.zeros(len(Z))
    Z = np.array(Z)
    Z = np.exp(Z)
    mp = Z/sum(Z)
    return mp
    

def outputtxt(posn, MODELPROB, MODELZ, MODELPARAM, MODELPARAMERR, MODELINFO, MODELMLE, fname):
    otpt = []
    MODELZ = list(MODELZ)
    header = "========================================================================================================== \n"
    header +="                                RESULTS FROM NESTED SAMPLING                                               \n"
    header +="========================================================================================================== \n"
    otpt.append(header)
    N = len(posn)
    lin1 = ','.join(str(i) for i in posn)
    lin1 = "The raw position data =  ["+lin1+"]  \n\n\n"
    otpt.append(lin1)
    linearr = ["","","",""]
    MODELS = ["(free, clean)","(pull, clean)","(free, mn)","(pull, mn)"]
    MODELZERR = [np.sqrt(MODELINFO[i]/N) for i in range(0, 4)]
    for i in range(0,4):
        linearr[i] = f"Model %d  %s  with Model Prob = %.3f \n"%(i+1,MODELS[i],MODELPROB[i])
        linearr[i] += f"ln Evidence = %.3f  +/- %.5f \n"%(MODELZ[i], MODELZERR[i])
        linearr[i] += f"Posterior Paramters \n"
        linearr[i] += f" 1. Diffusion Coefficient D_0 = %.3f  +/-  %.4f \n"%(MODELPARAM[i][0],3*MODELPARAMERR[i][0])
        linearr[i] += f" 2. Exponent alpha            = %.3f  +/-  %.4f \n"%(MODELPARAM[i][1],3*MODELPARAMERR[i][1])
        linearr[i] += f" 3. Applied Force f           = %.3f  +/-  %.4f \n"%(MODELPARAM[i][2],3*MODELPARAMERR[i][2])
        linearr[i] += f" 4. Var of Measurement noise  = %.3f  +/-  %.4f \n"%(MODELPARAM[i][3],3*MODELPARAMERR[i][3])
        linearr[i] += f" Maximum Likelihood = %.3f \n"%(MODELMLE[i])
        linearr[i] += f" Information H = %.5f  \n\n"%(MODELINFO[i])
    for i in range(0,4):
        otpt.append(linearr[i])
    hp = f"Highest Model probability for the given data is for Model %s \n\n\n\n"%(MODELS[MODELZ.index(max(MODELZ))])
    otpt.append(hp)
    outfile = open(fname, "w")
    outfile.writelines(otpt)
    outfile.close()
    
# def plotllgen(X, MODELPARAM,TRUEVAL):
#     time = X[0]
#     xtrue = ]
#     xm = X[2]
    
#     plotll(X,MODELPARAM,TRUEVAL)