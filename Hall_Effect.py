# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 12:48:46 2024

@author: Daniyaal and David
"""

from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

def linear(x, a, b):
    return a*x + b

def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

def plot_hall_vs_current(B, u_B, I, u_I, V, u_V):
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    fig.suptitle('Hall Voltage vs. DC Current for Magnetic Field {field} ± {unc} mT'.format(field=B, unc=u_B),
                 fontsize=16)
    
    popt, pcov = curve_fit(linear, I, V, sigma=u_V, absolute_sigma=True)
    a = popt[0]
    b = popt[1]
    
    chisquared = 0
    for i in range(0, 5):
        chisquared += ((V[i]-linear(I[i], a, b))**2)/((u_V[i])**2)
    chisquared_red = (1/(5 - 2))*chisquared
    print("Chi-Squared reduced for Field {field} mT is {chi}".format(field=B, chi=chisquared_red))
    
    goodness_of_fit = (1-chi2.cdf(chisquared_red,3))
    print("Goodness of fit for Field {field} mT is {good}".format(field=B, good=goodness_of_fit))
    
    pstd = np.sqrt(np.diag(pcov))
    print("{a}±{ua}".format(a=a, ua=pstd[0]))
    print("{b}±{ub}".format(b=b, ub=pstd[1]))

    
    axs[0].errorbar(I, V, ls='', marker='o', color='black', markersize=3, yerr=u_V, ecolor='red', label='Uncertainty')
    axs[0].plot(I, linear(I, a, b), linewidth=0.7, color='blue', label="Hall Voltage")
    axs[0].set_ylabel("Hall Voltage (Volts)", fontsize=15)
    axs[0].legend(loc=2, prop={'size': 8})
    
    axs[1].plot(I, V - linear(I, a, b), ls='', marker='o', markersize = 4, color='black')
    axs[1].axhline(y=0, color='blue')
    axs[1].errorbar(I, linear(I, a, b) - V, ls='', yerr=u_V, ecolor='red')
    axs[1].set_ylabel("Residuals", fontsize=15)
    axs[1].set_xlabel("DC Current (mA)", fontsize=18)
    
    return

def plot_electric_vs_density(B, u_B, I, u_I, V, u_V):
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    fig.suptitle('Electric Field vs. Current Density for Magnetic Field {field} ± {unc} mT'.format(field=B, unc=u_B),
                 fontsize=16)
    
    popt, pcov = curve_fit(linear, I, V, sigma=u_V, absolute_sigma=True)
    a = popt[0]
    b = popt[1]
    
    chisquared = 0
    for i in range(0, 5):
        chisquared += ((V[i]-linear(I[i], a, b))**2)/((u_V[i])**2)
    chisquared_red = (1/(5 - 2))*chisquared
    print("(E vs. J) Chi-Squared reduced for Field {field} mT is {chi}".format(field=B, chi=chisquared_red))
    
    goodness_of_fit = (1-chi2.cdf(chisquared_red,3))
    print("(E vs. J) Goodness of fit for Field {field} mT is {good}".format(field=B, good=goodness_of_fit))
    
    pstd = np.sqrt(np.diag(pcov))
    print("{a}±{ua}".format(a=a, ua=pstd[0]))
    print("{b}±{ub}".format(b=b, ub=pstd[1]))
    
    
    axs[0].errorbar(I, V, ls='', marker='o', color='black', markersize=3, yerr=u_V, ecolor='red', label='Uncertainty')
    # axs[0].plot(I, linear(I, a, b), linewidth=0.7, color='blue')
    axs[0].plot(I, linear(I, a, b), linewidth=0.7, color='blue', label="Electric Field")
    axs[0].set_ylabel("Electric Field (V/m)", fontsize=15)
    axs[0].legend(loc=2, prop={'size': 8})
    # axs[0].set_yticks(np.array([0.004, 0.006, 0.008, 0.010, 0.012, 0.014]))
    
    axs[1].plot(I, V - linear(I, a, b), ls='', marker='o', markersize = 4, color='black')
    axs[1].axhline(y=0, color='blue')
    axs[1].errorbar(I, linear(I, a, b) - V, ls='', yerr=u_V, ecolor='red')
    axs[1].set_ylabel("Residuals", fontsize=15)
    axs[1].set_xlabel("Current Density (A/m²)", fontsize=18)
    # axs[1].set_xticks(np.array([227, 256, 283, 310, 337]))
    return


def plot_hall_vs_field(B, u_B, I, u_I, V, u_V):
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    fig.suptitle('Hall Voltage vs. Magnetic Field for DC Current {current} ± {unc} mA'.format(current=I, unc=u_I),
                 fontsize=16)
    
    popt, pcov = curve_fit(linear, B, V, sigma=u_V, absolute_sigma=True)
    a = popt[0]
    b = popt[1]
    
    chisquared = 0
    for i in range(0, 3):
        chisquared += ((V[i]-linear(B[i], a, b))**2)/((u_V[i])**2)
    chisquared_red = (1/(3 - 2))*chisquared
    print("Chi-squared reduced for Current {current} mA is {chi}".format(current=I, chi=chisquared_red))
    
    goodness_of_fit = (1-chi2.cdf(chisquared_red,3))
    print("Goodness of fit for Current {current} mA is {good}".format(current=I, good=goodness_of_fit))
    
    pstd = np.sqrt(np.diag(pcov))
    print("{a}±{ua}".format(a=a, ua=pstd[0]))
    print("{b}±{ub}".format(b=b, ub=pstd[1]))
    
    axs[0].errorbar(B, V, ls='', marker='o', color='black', markersize=3, yerr=u_V, ecolor='red', label='Uncertainty')
    axs[0].plot(B, linear(B, a, b), linewidth=0.7, color='blue', label='Hall Voltage')
    axs[0].set_ylabel("Hall Voltage (Volts)", fontsize=15)
    axs[0].legend(loc=2, prop={'size': 8})
    
    axs[1].plot(B, V - linear(B, a, b), ls='', marker='o', markersize = 4, color='black')
    axs[1].axhline(y=0, color='blue')
    axs[1].errorbar(B, linear(B, a, b) - V, ls='', yerr=u_V, ecolor='red')
    axs[1].set_ylabel("Residuals", fontsize=15)
    axs[1].set_xlabel("Magnetic Field (mT)", fontsize=18)
    
    return

# Magnetic field
B_1 = 747
u_B = 20

B_2 = 722
B_3 = 702

# Current
I_1 = np.array([19.99, 22.54, 24.98, 27.52, 29.98])
u_I = np.array([0.02, 0.02, 0.02, 0.02, 0.02])

I_2 = np.array([20.02, 22.53, 25.04, 27.48, 29.97])
I_3 = np.array([19.99, 22.47, 25.06, 27.51, 30.00])

# Hall Voltage
V_1 = np.array([0.01, 0.011, 0.012, 0.013, 0.015])
u_V_1 = np.array([0.005, 0.003, 0.003, 0.003, 0.003])

V_2 = np.array([0.009, 0.01, 0.011, 0.012, 0.013])
u_V_2 = np.array([0.003, 0.004, 0.003, 0.003, 0.005])

V_3 = np.array([0.008, 0.01, 0.01, 0.012, 0.013])
u_V_3 = np.array([0.003, 0.003, 0.005, 0.005, 0.003])

B = np.array([B_1, B_2, B_3])
I = [I_1, I_2, I_3]
V = [V_1, V_2, V_3]
u_V = [u_V_1, u_V_2, u_V_3]

# The hall voltage for a fixed current:
V_c_1 = np.array([0.01, 0.009, 0.008])
u_V_c_1 = np.array([0.005, 0.003, 0.003])

V_c_2 = np.array([0.011, 0.01, 0.01])
u_V_c_2 = np.array([0.003, 0.004, 0.003])

V_c_3 = np.array([0.012, 0.011, 0.01])
u_V_c_3 = np.array([0.003, 0.003, 0.005])

V_c_4 = np.array([0.013, 0.012, 0.012])
u_V_c_4 = np.array([0.003, 0.003, 0.005])

V_c_5 = np.array([0.015, 0.013, 0.013])
u_V_c_5 = np.array([0.003, 0.005, 0.003])

V_c = [V_c_1, V_c_2, V_c_3, V_c_4, V_c_5]
u_V_c = [u_V_c_1, u_V_c_2, u_V_c_3, u_V_c_4, u_V_c_5]

for i in range(0, 3):
    plot_hall_vs_current(B[i], u_B, I[i], u_I, V[i], u_V[i])

I_c = [20.0, 22.5, 25.0, 27.5, 30.0]
u_I_c = [0.03, 0.04, 0.06, 0.02, 0.03]

for i in range(0, 5):
    plot_hall_vs_field(B, u_B, I_c[i], u_I_c[i], V_c[i], u_V_c[i])

# These are in metres
w = 1.27 * 10**(-3)
L = 13.24 * 10**(-3)
t = 6.9294 * 10**(-5) * 10**(-3)
A = t*w

E = [[], [], []]
J = [[], [], []]
u_E = [[], [], []]
for i in range(0, 3):
    E[i] = V[i]/w
    J[i] = (I[i]/A)
    u_E[i] = u_V[i]*(10**3)

for i in range(0, 3):
    plot_electric_vs_density(B[i], u_B, J[i], u_I, E[i], u_E[i])