#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 21:41:28 2019

@author: dauku
"""

# In[1]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use("ggplot")

# In[2]:
fig, ax = plt.subplots(figsize = (5, 3))
ax.set(xlim=(-3, 3), ylim = (-1, 1))

# In[3]:
x = np.linspace(-3, 3, 91)
t = np.linspace(1, 25, 30)
X2, T2 = np.meshgrid(x, t)

sinT2 = np.sin(2*np.pi*T2/T2.max())

# In[4]:
def costFunction(X, y, theta, lamda = 10):
    # Initializations
    m = len(y)
    J = 0
    
    # Computations
    h = X @ theta
    J_reg = (lamda / (2*m)) * np.sum(np.square(theta))
    J = float((1./(2*m)) * (h-y).T @ (h-y)) + J_reg;
    
    if np.isnan(J):
        return(np.inf)
        
    return(J)
    
def gradient_descent_reg(X, y, theta, alpha = 0.0005, lamda = 10, num_iters = 1000):
    # Initialisation of useful values
    m = np.size(y)
    J_history = np.zeros(num_iters)
    theta_0_hist, theta_1_hist = [], []
    
    for i in range(num_iters):
        # Hypothesis function
        h = np.dot(X, theta)
        
        # Calculating the grad function in vectorized form
        theta = theta - alpha * (1/m)* (  (X.T @ (h-y)) + lamda * theta )
        
        # Cost function in vectorized form
        J_history[i] = costFunction(X, y, theta, lamda)
        
        # Calculate the cost for each iteration (used to plot convergence)
        theta_0_hist.append(theta[0, 0])
        theta_1_hist.append(theta[1, 0])
        
    return theta, J_history, theta_0_hist, theta_1_hist

def closed_form_reg_solution(X, y, lamda = 10):
    m, n = X.shape
    I = np.eye((n))
    return np.linalg.inv(X.T @ X + lamda * I) @ X.T @ y

# In[5]:
"""Create Data"""
# creating the dataset
x = np.linspace(0, 1, 40)
noise = 1*np.random.uniform(size = 40)
y = np.sin(x * 1.5 * np.pi)
y_noise = (y + noise).reshape(-1, 1)
y_noise = y_noise - y_noise.mean()

# Plotting the data
plt.figure(figsize = (10, 6))
plt.scatter(x, y_noise, facecolors = "none", edgecolor = "darkblue", label = "sine + noise")



