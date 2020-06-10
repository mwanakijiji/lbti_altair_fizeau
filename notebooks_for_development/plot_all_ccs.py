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
lambda_over_D = pd.read_csv("./data/modern_contrast_curve.csv")
psf_profiles = pd.read_csv("./data/example_psf_profiles.csv", index_col=0)

# for lambda/B, there are a number of curves; we will read them all in
lambda_over_B_file_list = glob.glob("./data/lambda_B*w*planet*csv")

# read in PSF profiles
psf_profiles_rad_asec = 0.0107*np.arange(-0.5*len(psf_profiles["x_xsec_1"]),0.5*len(psf_profiles["x_xsec_1"]),step=1)

fig = plt.figure(figsize=(8,4))
# loop over all keys
for label, content in psf_profiles.items():
    # smooth the PSF profile and plot
    content_smoothed = gaussian_filter(content,sigma=2)
    plt.plot(psf_profiles_rad_asec,-2.5*np.log10(np.divide(content_smoothed,6e4)),
             alpha = 0.2, color="gray", linewidth=2)
plt.plot(lambda_over_D["rad_asec"],lambda_over_D["del_m_modern"],linewidth=4,
         label="$\lambda /D$ regime, based on fake planet injections")
for file_name in lambda_over_B_file_list:
    lambda_over_B = pd.read_csv(file_name)
    if (file_name == lambda_over_B_file_list[0]):
        # one name to the legend
        plt.plot(lambda_over_B["x"],lambda_over_B["y"],linewidth=4,color="red",
                label="$\lambda /B$ regime, based on KS test")
    else:
        plt.plot(lambda_over_B["x"],lambda_over_B["y"],linewidth=4,color="red")
        '''
        plt.plot(lambda_over_B["x"],lambda_over_B["y"],linewidth=4,
            label="$\lambda /B$ regime, based on KS test\n"+str(os.path.basename(file_name)))
        '''
plt.gca().invert_yaxis()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim([0,2.2])
plt.ylim([10,0])
plt.legend()
plt.ylabel("$\Delta$m", fontsize=18)
plt.xlabel("Radius (arcsec)", fontsize=18)
plt.tight_layout()
#plt.show()
plt.savefig("junk.pdf")
