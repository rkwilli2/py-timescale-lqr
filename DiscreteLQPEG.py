# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 11:15:45 2022

@author: davis
"""

import numpy as np
#import timescalecalculus as tsc
from numpy import transpose as tr
from numpy import array as mat
from numpy import linalg
import matplotlib.pyplot as plt
import scipy.stats as stats
import random

#Define Matrices
I = mat([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
x0 = mat([[-2],[.5],[3],[.5]])
A = mat([[2, 1, 0, 0],[0, 3.3, 0, 0],[0, 0, -2, -1],[0, 0, 0, 3.33]])
B = mat([[.75],[2],[0],[0]])
C = mat([[0],[0],[-1],[1.67]])

Q = I
R = mat([[1]])
W = mat([[1]])

F = ((B.dot(linalg.inv(R))).dot(tr(B))) - ((C.dot(linalg.inv(W))).dot(tr(C)))

#Set Trials
n=15

#Initialize and B.C.
s = I
S = list(mat([(s)]))

Kp = []

Ke = []

u = []

v = []
 
x = list(mat([x0]))

i = n-1

mu = 1.5

xp = [x0[0,0]]

xe = [x0[2,0]]
while i >= 0:
    kp=((((linalg.inv(R)).dot(tr(B))).dot(s)).dot(linalg.inv(I + mu*(F.dot(s))))).dot(I+mu*A)
    Kp.insert(0,kp)
    ke=((((linalg.inv(W)).dot(tr(C))).dot(s)).dot(linalg.inv(I + mu*(F.dot(s))))).dot(I+mu*A)
    Ke.insert(0,ke)
    z = A - (B.dot(kp)) + (C.dot(ke))
    s1 = mu*((tr(z)).dot(s))
    s2 = mu*(((I + mu*tr(z)).dot(s)).dot(z))
    s3 = (mu*((tr(kp).dot(R)).dot(kp))) - (mu*((tr(ke).dot(W)).dot(ke)))
    s = s + (mu*Q) + s1 + s2 +s3
    S.insert(0,s)
    i = i-1
    
for i in range (0,n):
    u.append(-Kp[i].dot(x[i]))
    v.append(Ke[i].dot(x[i]))
    xnew = ((I+(mu*A)).dot(x[i]) + mu*(B.dot(u[i])) + mu*(C.dot(v[i])))
    x.append(xnew)
    xp.append(xnew[0,0])
    xe.append(xnew[2,0])

iterations = []    
for i in range (0,n+1):
    iterations.append(i)
    
plt.figure(1)
plt.plot(iterations,xp,'b-^',label='Xp')
plt.plot(iterations,xe,'r-^',label='Xe')
#Blue is Estimation, Red is Sim
plt.xlabel('Iterations')
plt.ylabel('Values')
plt.title('Pursuer and Evader Positions')
plt.grid(True)
plt.legend()
plt.show()


