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

# make list of all the files
#lambda_over_D = pd.read_csv("./data/modern_contrast_curve.csv")
psf_profiles_all = pd.read_csv("./data/example_psf_profiles.csv", index_col=0)
# select subset
psf_profiles = psf_profiles_all.iloc[:, 0:10]

# lambda/D data
lambda_over_D = pd.read_csv("./data/modern_curve_20200713.csv")

# for lambda/B, there are a number of curves; we will read them all in
lambda_over_B = pd.read_csv("./data/lambda_B_cc_stripes_w_planets_avg_avg_20200714.csv")

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
         label="$\lambda /D$ regime (fake planet injections)")
# lambda/B data
plt.plot(lambda_over_B["x"],lambda_over_B["y"],linewidth=4,color="red",
        label="$\lambda /B$ regime (KS test)")

plt.gca().invert_yaxis()
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.xlim([0,2.2])
plt.ylim([11,0])
plt.legend(fontsize=17)
plt.ylabel("$\Delta$m", fontsize=23)
plt.xlabel("Radius (arcsec)", fontsize=23)
plt.tight_layout()
plt.show()
#plt.savefig("junk.pdf")
