from matplotlib.dates import MO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from Modelgen import Model
from Likelihood import Logl
from nestedsampling import NSalgo
from outputprocessor import *
from plotLikelihood import plotll

theta = [0.2,1,-0.3,0]
param = [1, 0, 100, 50, 1]

N = 1000
n = 100

DT = 1

p_data = np.array([50.0,43.991,45.551,40.801,44.942,30.352,41.731,32.839,23.453,24.174,26.658,42.995,29.819,26.953,26.038,31.555,20.682,26.771,20.865,19.459,18.254,5.244,12.406,15.52,2.616,10.983,7.248,8.92,10.981,14.72,12.3,-3.495,14.211,-0.833,17.996,-2.37,11.819,-4.138,-1.815,1.192,13.561,-0.511,1.985,12.103,5.668,11.2,2.832,0.478,2.659,10.321,8.485,19.542,8.341,14.542,9.954,3.898,10.348,19.923,8.019,11.035,8.833,9.886,14.906,18.788,21.279,0.027,15.448,7.89,7.049,7.346,14.273,10.35,12.234,2.728,1.864,13.076,4.534,7.99,-0.475,18.191,7.939,5.779,11.756,11.783,10.857,10.053,3.289,8.427,5.459,1.758,2.532,6.595,6.17,3.953,0.396,7.238,2.782,12.005,2.24,5.32])
x_true = np.array([50.0,45.439,41.854,47.473,41.123,32.771,32.738,30.643,27.686,30.542,35.337,35.453,27.37,22.701,27.14,28.004,27.167,24.252,19.938,16.23,13.946,14.425,14.182,12.04,9.999,9.23,9.008,9.263,11.093,10.901,7.988,5.055,5.024,6.092,6.879,5.183,4.366,2.393,3.372,4.219,6.009,7.71,8.923,8.809,5.643,3.518,4.411,4.462,4.94,4.908,6.775,10.276,11.263,10.242,9.659,8.873,6.83,9.671,9.189,9.008,10.722,12.376,12.869,14.473,14.179,9.772,11.843,11.019,8.049,7.354,6.517,6.716,4.403,3.383,3.729,6.409,7.362,5.851,6.821,9.302,8.752,9.447,8.693,8.803,8.364,6.978,5.88,4.417,4.538,3.058,2.525,2.125,1.533,3.152,4.85,6.234,7.486,10.634,6.71,3.312])
time = np.array([0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0,31.0,32.0,33.0,34.0,35.0,36.0,37.0,38.0,39.0,40.0,41.0,42.0,43.0,44.0,45.0,46.0,47.0,48.0,49.0,50.0,51.0,52.0,53.0,54.0,55.0,56.0,57.0,58.0,59.0,60.0,61.0,62.0,63.0,64.0,65.0,66.0,67.0,68.0,69.0,70.0,71.0,72.0,73.0,74.0,75.0,76.0,77.0,78.0,79.0,80.0,81.0,82.0,83.0,84.0,85.0,86.0,87.0,88.0,89.0,90.0,91.0,92.0,93.0,94.0,95.0,96.0,97.0,98.0,99.0])
# p_data = np.array([50.   , 51.367, 48.699, 44.642, 43.553, 49.184, 49.222, 50.955,       38.148, 46.938, 34.223, 38.415, 36.054, 35.257, 35.473, 29.603,       34.158, 28.346, 33.686, 31.855, 32.454, 36.521, 31.482, 35.63 ,       39.712, 38.944, 37.343, 39.255, 35.665, 31.73 , 28.254, 30.081,       35.46 , 34.247, 32.114, 30.253, 27.704, 27.316, 32.169, 32.617,       27.722, 32.482, 34.431, 28.408, 27.382, 28.881, 29.671, 29.079,       31.088, 33.544, 34.063, 39.901, 38.426, 49.951, 50.612, 54.932,       54.549, 60.98 , 60.88 , 59.873, 64.145, 65.389, 57.596, 49.069,       50.313, 49.782, 44.892, 56.176, 53.266, 52.5  , 43.84 , 38.461,       38.153, 38.319, 34.644, 32.851, 34.693, 31.565, 29.909, 31.77 ,       27.628, 24.255, 20.444, 24.825, 28.336, 30.011, 28.37 , 21.392,       17.319, 18.046, 16.723, 16.915, 19.138, 18.129, 17.593, 14.156,       16.958, 13.758, 17.603, 17.622])
domain = [[10**(-4),100],[-2,2],[-1,1],[0,100]]

# these are parameters fixed in each model
# 0 represents a variable parameter, 1 represents a fixed parameter
# for example [0,0,1,0] refers to free, clean
ff = np.array([[0,0,1,1],[0,0,0,1],[0,0,1,0],[0,0,0,0]])
nm = ff.shape[0]
MZ = np.zeros(nm)
MZE = np.zeros(nm)
MH = np.zeros(nm)
MP = np.zeros(ff.shape)
MPE = np.zeros(ff.shape)
MLE = np.zeros(4)
mprob = np.zeros(nm)

# saves to results to files with file name
filename = "LARGENOISEwithF.txt"
for i in range(0, nm):
    T = NSalgo(p_data, 100, domain, ff[i])
    MZ[i] = T[0]
    MZE[i] = T[1]
    MP[i] = T[2]
    MPE[i] = T[3]
    MH[i] = T[4]
    MLE[i] = T[5]
mprob = MODELPROB(MZ)

outputtxt(p_data, mprob,MZ,MP,MPE,MH,MLE, filename)
maxprob = max(mprob)
ind = list(mprob).index(maxprob)
infp = MP[ind]

plotll([time, p_data, x_true],infp,[0.2, 1, -0.3, 30])
    