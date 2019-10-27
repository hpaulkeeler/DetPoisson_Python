# K=funLtoK(L)
# The function funLtoK(L) converts a kernel L matrix into  a (normalized) 
# kernel K matrix. The K matrix has to be semi-positive definite.

import numpy as np #NumPy package for arrays, random number generation, etc

def funLtoK(L):
    eigenValuesL,eigenVectLK=np.linalg.eig(L); #eigen decomposition    
    eigenValuesK=eigenValuesL/(1+eigenValuesL); #eigenvalues of K
    eigenValuesK=np.diagflat(eigenValuesK); ##eigenvalues of L as diagonal matrix
    K=np.matmul(np.matmul(eigenVectLK,eigenValuesK),eigenVectLK.transpose()); #recombine from eigen components    
    K=np.real(K); #make sure all values are real        
    return K
    

#K=funLtoK(L)
# K=
#array([[ 0.76022099,  0.33480663, -0.09060773],
#       [ 0.33480663,  0.3320442 ,  0.32928177],
#       [-0.09060773,  0.32928177,  0.74917127]])
