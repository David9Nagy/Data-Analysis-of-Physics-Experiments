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
    return a*x**(1/2) + b

def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

def multdivpropagation(z, v, dv, w, dw):
    return z*((dv/v)+(dw/w))

def plot_delay_vs_lc(delay, u_delay, lc):
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    fig.suptitle('Length vs Period', fontsize=16)

    popt, pcov = curve_fit(linear, lc, delay, sigma=u_delay, absolute_sigma=True)
    a = popt[0]
    b = popt[1]

    chisquared = 0
    for i in range(0, 5):
        chisquared += ((delay[i]-linear(lc[i], a, b))**2)/((u_delay[i])**2)
    print("Chi squared =")
    print(chisquared)
    chisquared_red = (1/(5 - 2))*chisquared
    print("Chi-Squared reduced for transmission line is {chi}".format(chi=chisquared_red))
    
    goodness_of_fit = (1-chi2.cdf(chisquared_red,3))
    print("Goodness of fit for transmission line is {good}".format(good=goodness_of_fit))
    
    pstd = np.sqrt(np.diag(pcov))
    print("{a}±{ua}".format(a=a, ua=pstd[0]))
    print("{b}±{ub}".format(b=b, ub=pstd[1]))

    axs[0].errorbar(lc, delay, ls='', marker='o', color='black', markersize=3, yerr=u_delay, ecolor='red')
    axs[0].plot(lc, linear(lc, a, b), linewidth=0.7, color='blue')

#    axs[1].errorbar(lc, delay, ls='', color='black', markersize=3, yerr=u_delay, ecolor='red')
    axs[1].plot(lc, linear(lc, a, b) - delay, ls='', marker='o', markersize = 4, color='black')
    axs[1].axhline(y=0)
    axs[1].errorbar(lc, linear(lc, a, b) - delay, ls='', yerr=u_delay, ecolor='red')
    axs[1].set_ylabel("Length (cm)", fontsize=15, horizontalalignment='right', y=4.0)
    axs[1].set_xlabel("Period (s)", fontsize=18)

    return



# LC Units
lc = np.array([41, 36, 31, 26, 21, 16, 11, 6, 1])

# delay (micro secs)
length = np.array([34.5, 29.4, 43.9, 24.2, 19.2])
u_length = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
period = np.array([1.25, 1.07, 1.27, 0.96, 0.88])
# changing to seconds


# resistance
res = np.array([400, 400, 400, 400, 200, 600, 300, 300, 200])


plot_delay_vs_lc(length, u_length, period)