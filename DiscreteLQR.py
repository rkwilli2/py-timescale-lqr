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
x0 = mat([[7.1],[0],[0],[4.5]])
A = mat([[0, 1, 0, 0],[0, -2, 0, 0],[3, 0, 0, 0],[0, 0, 1, 0]])
B = mat([[0],[2],[0],[0]])
C = mat([[5, 0, 0, 0]])

Q = mat([[1]])
P = mat([[1]])
R = mat([[1]])
#Set Trials
n=30

#Initialize and B.C.
s = (tr(C).dot(P)).dot(C)
S = list(mat([(tr(C).dot(P)).dot(C)]))

K = []

u = []
 
x = list(mat([x0]))

i = n-1
while i >= 0:
    Kmidstep = ((tr(B).dot(s)).dot(B))+R
    K.insert(0,((linalg.inv(Kmidstep).dot(tr(B))).dot(s).dot(A)))
    Smidstep1 = ((tr(B).dot(s)).dot(B))+R
    Smidstep2 = s - (((s.dot(B)).dot(linalg.inv(Smidstep1))).dot(tr(B))).dot(s)
    s = (tr(A).dot(Smidstep2)).dot(A)+Q
    S.insert(0,s)
    print(s)
    i = i-1
    
for i in range (0,30):
    u.append(-K[i].dot(x[i]))
    x.append(A.dot(x[i]) + B.dot(u[i])) 
    print(i)

