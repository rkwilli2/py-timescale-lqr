# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 19:54:09 2023

@author: rkwil
"""

import numpy as np
import timescalecalculus as tsc
from numpy import transpose as tr
from numpy import array as mat
from numpy import linalg
import matplotlib.pyplot as plt
import scipy.stats as stats

# Define Variables
I = mat([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
x0 = mat([[-.5],[-1],[3],[2]])
A = mat([[2, 1, 0, 0],[0, 3, 0, 0],[0, 0, 2, -1],[0, 0, 0,-3]])
B = mat([[.75],[2],[0],[0]])
C = mat([[0],[0],[1],[2]])

Q = mat([[2,1,0,0],[0,1,0,0],[0,0,-2,-1],[0,0,0,1.5]])
R = mat([[1]])
W = mat([[1.5]])

s = mat([[1,1,0,0],[2,2,0,0],[0,0,1,2],[0,0,1.5,1.5]])

# Note, t0 and tf must be within the timescale
tf = 31
t0 = 0

ts = tsc.timescale([0,2,3,5,7,11,13,17,19,23,29,31], "TimeScale")

#Initialize and B.C.
S = list(mat([(s)]))

Kp = []

Ke = []

u = []

v = []
 
x = list(mat([x0]))

xp = [x0[0,0]]

xe = [x0[2,0]]

F = ((B.dot(linalg.inv(R))).dot(tr(B))) - ((C.dot(linalg.inv(W))).dot(tr(C)))

# Computation
domain = []
counter = 0
t = tf

while t >= t0:
    domain.insert(0, t)
    counter = counter + 1
    kp=((((linalg.inv(R)).dot(tr(B))).dot(s)).dot(linalg.inv(I + ts.mu(t)*(F.dot(s))))).dot(I+ts.mu(t)*A)
    Kp.insert(0,kp)
    ke=((((linalg.inv(W)).dot(tr(C))).dot(s)).dot(linalg.inv(I + ts.mu(t)*(F.dot(s))))).dot(I+ts.mu(t)*A)
    Ke.insert(0,ke)
    z = A - (B.dot(kp)) + (C.dot(ke))
    s1 = ts.mu(t)*((tr(z)).dot(s))
    s2 = ts.mu(t)*(((I + ts.mu(t)*tr(z)).dot(s)).dot(z))
    s3 = (ts.mu(t)*((tr(kp).dot(R)).dot(kp))) - (ts.mu(t)*((tr(ke).dot(W)).dot(ke)))
    s = s + (ts.mu(t)*Q) + s1 + s2 +s3
    S.insert(0,s)
    t = t - ts.mu(ts.rho(t))

for i in range(0, counter-1):
    u.append(-Kp[i].dot(x[i]))
    v.append(Ke[i].dot(x[i]))
    xnew = ((I+(ts.mu(ts.ts[i])*A)).dot(x[i]) + ts.mu(ts.ts[i])*(B.dot(u[i])) + ts.mu(ts.ts[i])*(C.dot(v[i])))
    x.insert(i, xnew)
    xp.insert(i, xnew[0,0])
    xe.insert(i, xnew[2,0])

print(len(xp))
plt.figure(1)
plt.plot(domain,xp,'b-^',label='Xp')
plt.plot(domain,xe,'r-^',label='Xe')
#Blue is Estimation, Red is Sim
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Pursuer and Evader Positions')
plt.grid(True)
plt.legend()
plt.show()