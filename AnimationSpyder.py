#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 21:41:28 2019

@author: dauku
"""

# In[1]:
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
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
plt.title("Noisy sine function, ploynomial ridge regression fit")
plt.xlabel("x")
plt.ylabel("y")

# Design matrix for polynomial regression of order 2
X = np.vstack((x, x**2)).T

# Calculating closed form regression coefficients with lambda = 0 and 25
theta_calc_0 = closed_form_reg_solution(X, y_noise, lamda = 0)
theta_calc_25 = closed_form_reg_solution(X,y_noise,lamda = 25)

# Plot 
plt.plot(x,X@theta_calc_0, '--', label = 'Ridge fit lambda = {}'.format(0))
plt.plot(x,X@theta_calc_25, '--', label = 'Ridge fit lambda= {}'.format(25))
plt.legend()
plt.show()

print('Coefficients with lambda = 25 are: ' ,theta_calc_25)

# In[]:
"Gradient Descent"
GDescent = gradient_descent_reg(X, y_noise, np.array([0, -5]).reshape(2, 1), alpha = 0.05, lamda = 25, num_iters = 1000)
GradientTheta = GDescent[0]
J_history_reg = GDescent[1]
theta_0 = GDescent[2]
theta_1 = GDescent[3]

# In[6]:
# Creat data for contour plot

l = 25 # Complexity hyperparameter lambda = 25

# Setup of meshgrid o theta values
T1, T2 = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-6, 3, 100))

# Computing the cost funtion for each theta combination
zs = np.array(  [costFunction(X, y_noise.reshape(-1,1),np.array([t1,t2]).reshape(-1,1),l) 
                     for t1, t2 in zip(np.ravel(T1), np.ravel(T2)) ] )

# Reshape the cost value
Z = zs.reshape(T1.shape)

# In[7]:

# plot the contour
fig1, ax1 = plt.subplots(figsize = (7, 7))
ax1.contour(T1, T2, Z, 100, cmap = "jet")
plt.show()
# Create animation
line, = ax1.plot([], [], "r", label = "Gradient Descent", lw = 1.5)
point, = ax1.plot([], [], "*", color = "red", markersize = 4)
value_display = ax1.text(0.02, 0.02, " ", transform = ax1.transAxes)

def init_1():
    line.set_data([], [])
    point.set_data([], [])
    value_display.set_text(" ")
    
    return line, point, value_display

def animate_1(i):
    # Animate line
    line.set_data(theta_0[:i], theta_1[:i])
    
    # Animate points
    point.set_data(theta_0[i], theta_1[i])
    
    # Animate value display
    value_display.set_text("Min = " + str(J_history_reg[i]))
    
    return line, point, value_display

ax1.legend(loc = 1)

anim1 = FuncAnimation(fig1, animate_1, init_func = init_1, frames = len(theta_0), interval = 100, repeat_delay = 60, blit = True)

plt.show()
# In[test]:
#Writer = matplotlib.animation.writers["ffmped"]
#writer = Writer(fps = 30, metadata = dict(artist = "David"), bitrate = 1800)
#anim1.save("/Users/dauku/Desktop/Python/Visualization/VisualizationExamples/Contour.mp4", writer = writer)

# In[8]:
"""3-D plot"""
fig2 = plt.figure(figsize = (7, 7))
ax2 = Axes3D(fig2)

# Surface plot
ax2.plot_surface(T1, T2, X, rstride = 5, cstride = 5, cmap = "jet", alpha = 0.5)
