# -*- coding: utf-8 -*-
"""

discreteLQPEG.py 

@author: Richard Williams
"""

import numpy as np
import timescalecalculus as tsc
from numpy import transpose as tr
from numpy import array as mat
from numpy import linalg
import matplotlib.pyplot as plt
import scipy.stats as stats

# Constants
# --------------------------------------------------------

# This identity should be the same size as A 

I = mat([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) # Assumes a 4x4 Identity

# --------------------------------------------------------

# LQPEG
# 
# Given parameters for a pursuit-evasion system, computes the state of a system
# using saddle-point strategies with the free final state case.
#
# The pursuit-evasion system is given by the following equations:
#
# State:
# x^\Delta (t) = Ax(t) + Bu(t) + Cv(t), x(t0) = x0
#
# Cost Functional:
# J(u, v) = 1/2 S(t_f) + 1/2 \int_{t0}^{tf} (x^T Q x + u^T R u - v^T W v)(tau)d(tau),
# S(t_f) = s
#
# @param ts                 - Time Scale (only discrete ones supported)
# @param A, B, C, Q, R, W   - Weighing matrices
# @param s                  - S(t_f)
# @param x0                 - x(t0)
# @param t0                 - Initial time
# @param tf                 - Final time
#
# @return - An array of values for x(t) for all values in the timescale that
#               fall in interval [t0, tf]
#
# Note: This algorithm relies on using indexes of lists to store values
# at discrete locations. More work would be required to allow support for
# dense locations in the time scale because of this approach. 
#
def lqpeg(ts, A, B, C, Q, R, W, s, x0, t0, tf):
    
    # Initializations
    # ------------------------------------------------
    
    # List of S Values
    S = list(mat([(s)]))
    
    # Initialize list of Kalman Gains
    Kp = []
    Ke = []
    
    # Temporary Control Vectors
    u = []
    v = []
    
    # State vectors for each player
    xp = []
    xe = []
    
    # List of values for x(t), the solution
    x = list(mat([x0]))
    
    # Compute commonly used term
    F = ((B.dot(linalg.inv(R))).dot(tr(B))) - ((C.dot(linalg.inv(W))).dot(tr(C)))
    
    # Begin Computation
    # -----------------------------------------
    
    domain = []     # Used for generating plots
    counter = 0     # Keeps track of how many values are in each list
    t = tf          # Start from tf
    
    # Iterate backwards from tf
    # This approach is needed because of the forward jump operators involved
    # in S(t) so we can properly calculate Kalman Gains
    
    while t != t0:
        domain.insert(0, t)
        counter = counter + 1
        
        # Calculate current Kalman Gains
        kp = ((((linalg.inv(R)).dot(tr(B))).dot(s)).dot(linalg.inv(I + ts.mu(t)*(F.dot(s))))).dot(I+ts.mu(t)*A)
        ke = ((((linalg.inv(W)).dot(tr(C))).dot(s)).dot(linalg.inv(I + ts.mu(t)*(F.dot(s))))).dot(I+ts.mu(t)*A)
        
        # Insert those into our list of Kalman Gain Values
        Kp.insert(0,kp)
        Ke.insert(0,ke)
        
        # Compute the value of "S sigma" for the next iteration
        z = A - (B.dot(kp)) + (C.dot(ke))
        s1 = ts.mu(t)*((tr(z)).dot(s))
        s2 = ts.mu(t)*(((I + ts.mu(t)*tr(z)).dot(s)).dot(z))
        s3 = (ts.mu(t)*((tr(kp).dot(R)).dot(kp))) - (ts.mu(t)*((tr(ke).dot(W)).dot(ke)))
        s = s + (ts.mu(t)*Q) + s1 + s2 +s3
        
        # Save this value
        S.insert(0,s)
        
        # Apply a backward jump. If tf > t0, this t will eventually reach t0.
        # Note rho(t) = t when the timescale is left-dense. This would need
        #    to be addressed when extending this algorithm to all time scales.
        t = ts.rho(t) 
    
    # Uses proven theorem that Bu + Cv = -BKp x + CKe x for saddle-point strategies
   
    # We can use iterations starting from t0 to tf to compute each subsequent value
    # of x(t) by using the relationship x^\Delta = Ax + Bu + Cv
    
    for i in range(0, counter):
        # Compute the optimal strategies for u and v, put them in the list
        u.append(-Kp[i].dot(x[i]))
        v.append(Ke[i].dot(x[i]))
        
        # Compute x sigma then insert it into the array of values
        # x^\Delta = Ax + Bu + Cv
        # x sigma - x = mu (Ax + Bu + Cv), by definition of delta derivative for time scales
        # x sigma = x + muAx + muBu + muCv
        # x sigma = (I + muA)x + mu(Bu + Cv)
        
        mu = ts.mu(ts.ts[i])
        xSigma = (I + mu * A).dot(x[i]) + mu*(B.dot(u[i]) + C.dot(v[i]))
        x.insert(i, xSigma)
        
    # Now that x is populated, we return
    return x
    
# Driver Code

# Define Variables
x0 = mat([[2],[1],[1],[2]])
A = mat([[2, 0, 0, 0],[0, 1, 0, 0],[0, 0, 3, 1],[0, 0, 1, 1]])
B = mat([[1],[3],[0],[0]])
C = mat([[0],[0],[2],[-2]])

Q = mat([[1,1,0,0],[1,4,0,0],[0,0,1,-1],[0,0,-1,.1]])
R = mat([[1]])
W = mat([[1.3]])

s = mat([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])

# Note, t0 and tf must be within the timescale
tf = 10
t0 = 2

myTS = [2, 3, 4, 5, 6, 6.15, 6.3, 6.45, 6.6, 6.75, 6.9, 7.05, 7.2, 7.35, 7.5, 7.65, 7.8, 7.95, 8, 9, 10]
#ts = tsc.integers(0, 20)
ts = tsc.timescale(myTS, "TimeScale")

x = lqpeg(ts, A, B, C, Q, R, W, s, x0, t0, tf)
print(x)
