# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:11:45 2024

@author: Daniyaal and David
"""

from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

def wavelength(m, a):
    return (m/d_0)*a

def dsound(m, d):
    return (m*wavelength)/d

def practice_plot(m, theta, u_theta):
    sin_theta = np.sin(np.deg2rad(theta))
    u_sin_theta = np.sin(np.deg2rad(u_theta))
    
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    fig.suptitle('Sine of Angle vs. Diffraction Order for 2500 Lines/Inch Grating', fontsize=16)

    popt, pcov = curve_fit(wavelength, m, sin_theta, sigma=u_sin_theta, absolute_sigma=True, maxfev=1000)
    a = popt[0]

    chisquared = 0
    for i in range(0, len(theta)):
        chisquared += ((sin_theta[i]-wavelength(m[i], a))**2)/((u_sin_theta[i])**2)
    chisquared_red = (1/(len(theta) - len(popt)))*chisquared
    print("Chi-Squared reduced for 2500 Lines/inch grating is {chi}".format(chi=chisquared_red))
    
    goodness_of_fit = (1-chi2.cdf(chisquared_red,3))
    print("Goodness of fit for 2500 Lines/inch grating is {good}".format(good=goodness_of_fit))
    
    pstd = np.sqrt(np.diag(pcov))
    print("Wavelength: {a} ± {ua}".format(a=a, ua=pstd[0]))

    axs[0].errorbar(m, sin_theta, ls='', marker='o', color='black', markersize=3, yerr=u_sin_theta, ecolor='red', label='Uncertainty')
    axs[0].plot(m, wavelength(m, a), linewidth=0.7, color='blue', label='Expected Sin Theta Linear Regression')
    axs[0].legend(loc=1, prop={'size': 7})

    axs[1].plot(m, wavelength(m, a) - sin_theta, ls='', marker='o', markersize = 3, color='black')
    axs[1].axhline(y=0, color='blue', linewidth=0.7)
    axs[1].errorbar(m, wavelength(m, a) - sin_theta, ls='', yerr=u_sin_theta, ecolor='red')
    axs[1].set_ylabel("Sine of Angle", fontsize=15, horizontalalignment='right', y=4.0)
    axs[1].set_xlabel("Diffraction Order", fontsize=18)
    axs[1].set_xticks([-2, -1, 0, 1, 2])
    
    print()
    
    return a

def plot(m, theta, u_theta, f):
    sin_theta = np.sin(np.deg2rad(theta))
    u_sin_theta = np.sin(np.deg2rad(u_theta))
    
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    fig.suptitle('Sin(Angle) vs. Diffraction Order for Ultrasonic Wave (frequency = {f} MHz)'.format(f = f), fontsize=16)

    popt, pcov = curve_fit(dsound, m, sin_theta, sigma=u_sin_theta, absolute_sigma=True, maxfev=1000)
    d = popt[0]

    chisquared = 0
    for i in range(0, len(theta)):
        chisquared += ((sin_theta[i]-dsound(m[i], d))**2)/((u_sin_theta[i])**2)
    chisquared_red = (1/(len(theta) - len(popt)))*chisquared
    print("Chi-Squared reduced for Ultrasonic Standing Wave grating of frequency {f} MHz is {chi}".format(chi=chisquared_red, f = f))
    
    goodness_of_fit = (1-chi2.cdf(chisquared_red,3))
    print("Goodness of fit for Ultrasonic Standing Wave grating of frequency {f} MHz is {good}".format(good=goodness_of_fit, f = f))
    
    pstd = np.sqrt(np.diag(pcov))
    print("Wavelength: {a} ± {ua}".format(a=d, ua=pstd[0]))

    axs[0].errorbar(m, sin_theta, ls='', marker='o', color='black', markersize=3, yerr=u_sin_theta, ecolor='red', label='Uncertainty')
    axs[0].plot(m, dsound(m, d), linewidth=0.7, color='blue', label='Expected Sin Theta Linear Regression')
    axs[0].legend(loc=1, prop={'size': 7})

    axs[1].plot(m, dsound(m, d) - sin_theta, ls='', marker='o', markersize = 3, color='black')
    axs[1].axhline(y=0, color='blue', linewidth=0.7)
    axs[1].errorbar(m, dsound(m, d) - sin_theta, ls='', yerr=u_sin_theta, ecolor='red')
    axs[1].set_ylabel("Sine of Angle", fontsize=15, horizontalalignment='right', y=4.0)
    axs[1].set_xlabel("Diffraction Order", fontsize=18)
    axs[1].set_xticks([-2, -1, 0, 1, 2])
    
    print()
    
    return d

m = np.array([-2, -1, 0, 1, 2])
theta_0 = np.array([6.8375, 3.4125, 0, -3.4000, -6.8275])
d_0 = (1/2500)/39.37
u_theta_0 = np.array([0.3, 0.3, 0.4, 0.3, 0.3])

wavelength = practice_plot(m, theta_0, u_theta_0)

f_s = np.array([2.004, 1.951, 1.904, 1.852, 1.802])
u_f_s = np.array([])

u_theta = np.array([0.02, 0.02, 0.02, 0.02, 0.02])
theta_1 = np.array([0.1100, 0.0625, 0, -0.0400, -0.0850])
theta_2 = np.array([0.0900, 0.0400, 0, -0.0575, -0.1025])
theta_3 = np.array([0.0850, 0.0425, 0, -0.0525, -0.1100])
theta_4 = np.array([0.0825, 0.0350, 0, -0.0400, -0.0750])
theta_5 = np.array([0.0500, 0.0375, 0, -0.0275, -0.0725])

d1 = plot(m, theta_1, u_theta, f_s[0])
d2 = plot(m, theta_2, u_theta, f_s[1])
d3 = plot(m, theta_3, u_theta, f_s[2])
d4 = plot(m, theta_4, u_theta, f_s[3])
d5 = plot(m, theta_5, u_theta, f_s[4])