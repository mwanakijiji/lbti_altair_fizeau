# This reads in contrast curve and PSF profile info for making a plot

# Created 2020 June 5 by E.S.

import glob
import os
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter

# make list of all the files
#lambda_over_D = pd.read_csv("./data/modern_contrast_curve.csv")
psf_profiles_all = pd.read_csv("./data/example_psf_profiles.csv", index_col=0)
# select subset
psf_profiles = psf_profiles_all.iloc[:, 0:10]

# lambda/D data
lambda_over_D = pd.read_csv("./data/modern_curve_20200713.csv")

# for lambda/B, there are a number of curves; we will read them all in
lambda_over_B_N = pd.read_csv("../results_20201120/lambda_B_cc_stripe_w_planet_20201021_allN.csv")
lambda_over_B_E = pd.read_csv("../results_20201120/lambda_B_cc_stripe_w_planet_20201021_allE.csv")
lambda_over_B_S = pd.read_csv("../results_20201120/lambda_B_cc_stripe_w_planet_20201021_allS.csv")
lambda_over_B_W = pd.read_csv("../results_20201120/lambda_B_cc_stripe_w_planet_20201021_allW.csv")

# read in PSF profiles
psf_profiles_rad_asec = 0.0107*np.arange(-0.5*len(psf_profiles["x_xsec_1"]),0.5*len(psf_profiles["x_xsec_1"]),step=1)

fig = plt.figure(figsize=(8,6))
# loop over all keys
for label, content in psf_profiles.items():
    # smooth the PSF profile and plot
    content_smoothed = gaussian_filter(content,sigma=2)
    plt.plot(psf_profiles_rad_asec,-2.5*np.log10(np.divide(content_smoothed,6e4)),
             alpha = 0.2, color="gray", linewidth=2)
# lambda/D data
plt.plot(lambda_over_D["rad_asec"],lambda_over_D["del_m_modern"],linewidth=4,
         label="$\lambda /D$ regime\n(planet injection and S/N measurement)")

# prune lambda/B data
# r > 0.42 asec: KS test fails because PSF is within 1 lambda/D from edge
# del_m < 3.722: the 4 models do not all simultaneously have a solution
#import ipdb; ipdb.set_trace()

# lambda/B data: all mags as dotted lines
plt.plot(lambda_over_B_N["x"].where(lambda_over_B_N["x"] < 0.42),lambda_over_B_N["y"].where(lambda_over_B_N["x"] < 0.42),linewidth=4,
        linestyle = ":")
plt.plot(lambda_over_B_E["x"].where(lambda_over_B_E["x"] < 0.42),lambda_over_B_E["y"].where(lambda_over_B_E["x"] < 0.42),linewidth=4,
        linestyle = ":")
plt.plot(lambda_over_B_S["x"].where(lambda_over_B_S["x"] < 0.42),lambda_over_B_S["y"].where(lambda_over_B_S["x"] < 0.42),linewidth=4,
        linestyle = ":")
plt.plot(lambda_over_B_W["x"].where(lambda_over_B_W["x"] < 0.42),lambda_over_B_W["y"].where(lambda_over_B_W["x"] < 0.42),linewidth=4,
        linestyle = ":")
# lambda/B data: overplot solid lines where all 4 models are valid: i.e., where abs_mag > 3.772, and we need to subtract
# off set point of 1.70 to get corresponding del_m (abs_mag = del_m + 1.70)
plt.plot(lambda_over_B_N["x"].where(np.logical_and(lambda_over_B_N["x"] < 0.42,lambda_over_B_N["y"] > np.subtract(3.772,1.7))),
        lambda_over_B_N["y"].where(np.logical_and(lambda_over_B_N["x"] < 0.42,lambda_over_B_N["y"] > np.subtract(3.772,1.7))),linewidth=4,
        color="#ff7f0e", label="$\lambda /B$ regime, N\n(planet injection and KS test)")
plt.plot(lambda_over_B_E["x"].where(np.logical_and(lambda_over_B_E["x"] < 0.42,lambda_over_B_E["y"] > np.subtract(3.772,1.7))),
        lambda_over_B_E["y"].where(np.logical_and(lambda_over_B_E["x"] < 0.42,lambda_over_B_E["y"] > np.subtract(3.772,1.7))),linewidth=4,
        color="#2ca02c", label="$\lambda /B$ regime, E")
plt.plot(lambda_over_B_S["x"].where(np.logical_and(lambda_over_B_S["x"] < 0.42,lambda_over_B_S["y"] > np.subtract(3.772,1.7))),
        lambda_over_B_S["y"].where(np.logical_and(lambda_over_B_S["x"] < 0.42,lambda_over_B_S["y"] > np.subtract(3.772,1.7))),linewidth=4,
        color="#d62728", label="$\lambda /B$ regime, S")
plt.plot(lambda_over_B_W["x"].where(np.logical_and(lambda_over_B_W["x"] < 0.42,lambda_over_B_W["y"] > np.subtract(3.772,1.7))),
        lambda_over_B_W["y"].where(np.logical_and(lambda_over_B_W["x"] < 0.42,lambda_over_B_W["y"] > np.subtract(3.772,1.7))),linewidth=4,
        color="#9467bd", label="$\lambda /B$ regime, W")

# LOGARITHMIC X-AXIS
plt.gca().invert_yaxis()
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.xlim([0.1,2.2])
ax = plt.gca()
ax.set_xscale('log')
plt.tick_params(axis='x', which='minor', labelsize=17)
ax.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
plt.ylim([11,0])
plt.legend(fontsize=17)
plt.ylabel("$\Delta$m", fontsize=23)
plt.xlabel("Radius (arcsec)", fontsize=23)
plt.tight_layout()
plt.show()
#plt.savefig("junk.pdf")

'''
# LINEAR X-AXIS
plt.gca().invert_yaxis()
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.xlim([0,2.2])
plt.ylim([11,0])
plt.legend(fontsize=17)
plt.ylabel("$\Delta m$", fontsize=23)
plt.xlabel("Radius (arcsec)", fontsize=23)
plt.tight_layout()
plt.show()
#plt.savefig("junk.pdf")
'''
