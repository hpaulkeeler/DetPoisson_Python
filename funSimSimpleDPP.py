# function indexConfig=funSimSimpleDPP(eigenVectorsL,eigenValuesL)
# This function simulates a determinantal point process (DPP) provided a
# L matrix. It was used to produce the results in the paper by Blaszczyszyn 
# and Keeler[1].
#
# The algorithm exists in Kulesza and Taskar[2]; see Algorithm 1.
# Also see Algorithm 1 in Lavancier, Moller and Rubak[3].
#
# If you use this code in a publication, please cite the paper by
# Blaszczyszyn and Keeler[1].
#
# Author: H.P. Keeler, Inria/ENS, Paris, and University of Melbourne,
# Melbourne, 2018
#
# References:
# [1] Blaszczyszyn and Keeler, Determinantal thinning of point processes
# with network learning applications, 2018.
# [2] Kulesza and Taskar, "Determinantal point processes for machine 
# learning",Now Publisers, 2012
# [3] Lavancier, Moller and Rubak, "Determinantal point process models and
# statistical inference", Journal of the Royal Statistical Society --
# Series B, 2015.

#import relevant libraries
import numpy as np
from scipy.linalg import orth

def funSimSimpleDPP(eigenVectorsL,eigenValuesL):
    
    #START - Sampling/simulating DPP - START
    # START Simulating/sampling DPP
    eigenValuesK = eigenValuesL / (1+eigenValuesL); #eigenvalues of K
    indexEig = (np.random.rand(eigenValuesK.size) < eigenValuesK );#Bernoulli trials
    
    #number of points in the DPP realization
    numbPointsDPP= np.sum(indexEig);  #number of points 
    #retrieve eigenvectors corresponding to successful Bernoulli trials
    spaceV = eigenVectorsL[:, indexEig]; #subspace V
    indexDPP=np.zeros(numbPointsDPP,dtype='int'); #index for final DPP configuration
    
    #Loop through for all points
    for ii in range(numbPointsDPP):
        #Compute probabilities for each point i    
        Prob_i = np.sum(spaceV**2, axis=1);#sum across rows
        Prob_i = np.cumsum(Prob_i/ np.sum(Prob_i)); #normalize
        
        #Choose a new point using PMF Prob_i  
        indexCurrent=(np.random.rand() <= Prob_i).argmax();
        indexDPP[ii]=indexCurrent;    
        
        #Choose a vector to eliminate
        jj = (np.abs(spaceV[indexCurrent, :]) > 0).argmax() 
        columnVj = spaceV[:, jj];
        
        #Update matrix V by removing Vj component from the space
        spaceV = spaceV- (np.outer(columnVj,(spaceV[indexCurrent, :] / columnVj[indexCurrent]))); 
        #Orthonormalize (using singular value decomposition - could also use qr)
        spaceV = orth(spaceV);
    
    #Loop finished   
    indexDPP=np.sort(indexDPP); #sort points
    #END - Simulating/sampling DPP - END
    return indexDPP;
