# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 12:18:45 2017

@author: jdkern
"""
from __future__ import division
import numpy as np

#Define state vector
# 1 - sunny; 2 - rainy

x = np.array([0,1])
P = np.array(([.9,.1],[.5,.5]))
a = np.array([0,0])

# Probability of weather 10 days from now
x_n = np.zeros((10,2))

for i in range(1,11):
    for j in range(1,i+1):
        if j<2:
            a = np.dot(x,P)
        else: 
            a = np.dot(a,P)
    
    x_n[i-1,:] = a


# Steady state probability

# q (P - I) = 0

P_I = P - np.array(([1,0],[0,1]))

# Solve system of equations 
# -.1(q1) + .1(q2) = 0
#  .5(q1) - .5(q2) = 0
#   q1 + q2 = 1

#% -.1(q1) + .1(1-q1) = 0; --> -0.1q1 + 0.1 - 0.1q1 = 0 --> 0.2q1 = 0.1
#% 0.5(q1)-0.5(1-q2) = 0;
#% 

q1 = 0.1/0.2
q2 = 1-q1

sunny = q1*365
rainy = q2*365

#Class Example
x1 = np.array([0,0,1])
P1 = np.array(([0.8, 0.1, 0.1],[0.6, 0.2, 0.2],[0.5, 0.25, 0.25]))
a1 = np.array([0,0,0])

x_n1 = np.zeros((10,3))

for i in range(1,11):
    for j in range(1,i+1):
        if j<2:
            a1 = np.dot(x1,P1)
        else: 
            a1 = np.dot(a1,P1)
            
    x_n1[i-1,:] = a1
    
#the probabilities reach steady state after ~ 5 generations

P_I2 = P1 - np.array(([1,0,0],[0,1,0],[0,0,1]))

A = np.array(([-.2,.6,.5],[.1,-.8,.25],[.85,.95,0]))
B = np.array(([0,0,.75]))
X= np.linalg.solve(A,B)


#Does it make sense to do this? How should we interpret the results?

# It makes some sense to do this, because the results indicate that a
# person is more likely to have a job as a professional than a skilled
# or unskilled laborer, assuming that the percent likelihood of each of 
# the jobs remains the same for future generations of each profession. By definition, 
# the Markov chain can make predictions about a daughter's job based solely 
# on the mother's job. Since the daughters of each mother will work in a 
# slightly different time, the economy and job availability will be different
# than that of the mother's time. For this reason, the results are only true 
# if the mother and the daughter live in a steady state economy, where the 
# proportions of professional, skilled labor, and unskilled labor jobs are 
# steady, and the population remains proportional to the job availabilities. 

