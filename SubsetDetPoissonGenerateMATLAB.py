# This simulates determinatally-thinned point processes that have been 
# fitted to thinned-point process based on the method outlined in the paper 
# by Blaszczyszyn and Keeler[1], which is essentially the method developed 
# by Kulesza and Taskar[2]. It then gathers empirical estimates of nearest
# neighbour distributions and contact distributions; for details see,
# the books by Chiu, Stoyan, Kendall, and Mecke[3] or Baddeley, Rubak and
# Turner[4]. It also calculates these distributions using determinant
# equations by simulating the underlying Poisson point process, and not the
# determinatally-thinned point process; see Blaszczyszyn and Keeler[1],
# Section III. Part of this is based on the (reduced) Palm distribution of
# determinantal point processes derived by Shirai and Takahashi[5]
#
# This is the third file (of three files) to run to reproduce the results
# presented in the paper by Blaszczyszyn and Keeler[1].
#
# The data used for fitting (or training) is stored in the file Subset.mat,
# which is generated with the MATLAB file SubsetGenerate.m.
#
# The fitting paramters are stored locally in the file SubsetFitParam.npz
#
# REQUIREMENTS:
# Uses Statistics (and Machine learning) Toolbox for random variable.
#
# Author: H.P. Keeler, Inria/ENS, Paris, and University of Melbourne,
# Melbourne, 2018
#
# References:
# [1] Blaszczyszyn and Keeler, Determinantal thinning of point processes
# with network learning applications, 2018.
# [2] Kulesza and Taskar, "Determinantal point processes for machine
# learning",Now Publisers, 2012
# [3] Chiu, Stoyan, Kendall, and Mecke, "Stochastic geometry and its
# applications", Wiley.
# [4] Baddeley, Rubak and Turner, "Spatial point patterns: Methodology and
# applications with R, 2016.
# [5] Shirai and Takahashi, "Random point fields associated with certain
# Fredholm determinants I -- fermion, poisson and boson point", 2003.12


import numpy as np; #NumPy package for arrays, random number generation, etc
import matplotlib.pyplot as plt #for plotting
from matplotlib import collections  as mc #for plotting line segments
from scipy.io import loadmat #for reading mat files
from scipy.optimize import minimize #For optimizing
from scipy.stats import poisson #for the Poisson probability mass function

from funNeighbourL import funNeighbourL
from funSimSimpleDPP import funSimSimpleDPP

plt.close("all"); # close all figures

numbSim=10**3;

###START Load up values from MATLAB .mat file START###
dataMATLAB=loadmat('Subset.mat');
lambda0=np.double(dataMATLAB['lambda']);#intensity of underlying Poisson PP
xx0=np.double(dataMATLAB['xx0']); 
yy0=np.double(dataMATLAB['yy0']); 
areaSample=np.double(dataMATLAB['areaSample']); #area of sample window
rSub=np.double(dataMATLAB['rSub']); #radius of matern or triangular process
lambdaSub=np.double(dataMATLAB['lambdaSub']); #intensity of subset PP
windowSample=dataMATLAB['windowSample'][0]; #vector describing window dims
choiceModel=np.int(dataMATLAB['choiceModel']); #model number (ie 1,2 or 3)
labelModel=str(dataMATLAB['labelModel'][0]); #name/label of model
booleDisk=np.int(dataMATLAB['booleDisk'])!=0; #if simulation window is disk
#x/y values of all underlying Poisson PPs
ppStructTemp=dataMATLAB['ppStructPoisson']; 
numbTrain=ppStructTemp.size; #total number of simulations
#extract data for underlying Poisson point processes
xxList=[np.concatenate(ppStructTemp[ss][0][0]) for ss in range(numbTrain)];
yyList=[np.concatenate(ppStructTemp[ss][0][1]) for ss in range(numbTrain)];
ppXYPoisson=[(xxList[ss],yyList[ss])for ss in range(numbTrain)];

nList=[np.int(ppStructTemp[ss][0][2]) for ss in range(numbTrain)];
nArray=np.array(nList);
#extract data for subset point processes
indexSubTemp=dataMATLAB['indexCellSub'];  
indexListSub=[np.array(np.concatenate(indexSubTemp[ss][0])-1,dtype=int) for ss in range(numbTrain)];
#NOTE: need to subtract one from MATLAB indices as Python indexing starts at zero.

###END Load up values from MATLAB .mat file END###

###START Load up values from MATLAB SubsetFitParam.npz file START###

fileVarsFitted=np.load('SubsetFitParam.npz'); #load file

thetaMax=fileVarsFitted['thetaMax'];
T=fileVarsFitted['T'];
sigma=fileVarsFitted['sigma'];
N=fileVarsFitted['N'];
M=fileVarsFitted['M'];
choiceModelFitted=fileVarsFitted['choiceModelFitted'];
booleOptSigma=fileVarsFitted['booleOptSigma'];
choiceKernel=fileVarsFitted['choiceKernel'];

###END Load up values from MATLAB SubsetFitParam.npz file END###

if (numbSim+T>numbTrain):
    raise SystemExit('Need to create more realziations with SubsetGenerate.m');
else:
    #Look at unused realizations (ie the ones not used for fitting)
    #select a random subset of unused realizations
    ttValuesPerm=np.arange(T,numbSim+T);
    np.random.shuffle(ttValuesPerm);
    ttValues=ttValuesPerm[np.arange(numbSim)];


#Initiate variables for collecting statistics
numbSub=np.zeros(numbSim); #number of points in subset point process
numbDPP=np.zeros(numbSim); #number of points in detrminantal point process

for ss in range(numbSim):
    tt=ttValues[ss];
    xxPoisson=ppXYPoisson[tt][0];
    yyPoisson=ppXYPoisson[tt][1];
    indexSub=indexListSub[tt]; #index for sub point process    
    numbSub[ss]=indexSub.size; 
    
    #generate L matrix based on Poisson point realization
    L=funNeighbourL(xxPoisson,yyPoisson,lambda0,choiceKernel,sigma,thetaMax,N,M);
    #Eigen decomposition
    eigenValuesL, eigenVectorsL=np.linalg.eig(L);
        
    #Simulate next DPP generation
    indexDPP=funSimSimpleDPP(eigenVectorsL,eigenValuesL);
    numbDPP[ss]=indexDPP.size;
    
#x/y values of subset point process
xxSub=xxPoisson[indexDPP]; yySub=yyPoisson[indexDPP];

lambdaEmpDPP=np.mean(numbDPP)/areaSample #empirical intensity of DPP
lambdaEmpSub=np.mean(numbSub)/areaSample #empirical intensity of subset PP
    
#Plotting 
#Plot Poisson point process
plt.scatter(xxPoisson,yyPoisson, edgecolor='k', facecolor='none');
plt.xlabel("x"); plt.ylabel("y");
#random color vector
vectorColor=(np.asscalar(np.random.rand(1)), np.asscalar(np.random.rand(1)), np.asscalar(np.random.rand(1)));
#Plot determinantally-thinned Poisson point process
plt.scatter(xxSub,yySub,edgecolor='none',facecolor=vectorColor);

