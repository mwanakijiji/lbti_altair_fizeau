#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This reads in contrast curve and PSF profile info for making a plot

# Created 2020 June 5 by E.S.


# In[24]:


import glob
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from astropy.io import fits

#%matplotlib inline
get_ipython().run_line_magic('matplotlib', 'qt')


# In[13]:


# make list of all the files

lambda_over_D = pd.read_csv("./data/modern_contrast_curve.csv")
lambda_over_B = pd.read_csv("./data/lambda_B_cc.csv")
psf_profiles = pd.read_csv("./data/example_psf_profiles.csv", index_col=0)


# In[14]:


psf_profiles_rad_asec = 0.0107*np.arange(-0.5*len(psf_profiles["x_xsec_1"]),0.5*len(psf_profiles["x_xsec_1"]),step=1)


# In[27]:


# loop over all keys
for label, content in psf_profiles.items():
    # smooth the PSF profile and plot
    content_smoothed = gaussian_filter(content,sigma=2)
    plt.plot(psf_profiles_rad_asec,-2.5*np.log10(np.divide(content_smoothed,6e4)), 
             alpha = 0.2, color="gray", linewidth=2)

plt.plot(lambda_over_D["rad_asec"],lambda_over_D["del_m_modern"],linewidth=4,
         label="$\lambda /D$ regime, based on fake planet injections")
plt.plot(lambda_over_B["x"],lambda_over_B["y"],linewidth=4,
         label="$\lambda /B$ regime, based on KS test")
plt.gca().invert_yaxis()
plt.xlim([0,2.2])
plt.ylim([10,0])
plt.legend()
plt.ylabel("$\Delta$m")
plt.xlabel("Radius (arcsec)")
plt.show()

