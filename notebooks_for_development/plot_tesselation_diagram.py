import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from scipy import ndimage
from astropy.io import fits
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.cm as cmx

#%matplotlib inline
#%matplotlib qt

# read in a pre-existing mask and remove slices to make a truncated mask

# lambda/D
test_data_frame, hdr = fits.getdata("data/mask_406x406_rings_4quad_fits_coarse_20200409.fits", 0, header=True)

# lambda/B

this_cube = np.copy(test_data_frame)
fig, ax = plt.subplots(1, 1)
lower_lim = -2.1721
upper_lim = 2.1721

# plot the grey (not considered) regions

median_img = np.nanmedian(this_cube,axis=0)
plt.imshow(median_img, cmap="gray", vmin=-1, vmax=0.2,
           extent=[lower_lim,upper_lim,lower_lim,upper_lim], origin="lower")

# plot the white (considered) regions
for plot_num in range(0,len(this_cube)):
    this_plot = this_cube[plot_num,:,:].astype('float')
    this_plot[this_plot == 0] = np.nan # this to a
    plt.imshow(this_plot, cmap="gray", vmin=-3, vmax=1,
               extent=[lower_lim,upper_lim,lower_lim,upper_lim], origin="lower")
    plt.contour(this_plot, levels=[1], colors="k", extent=[lower_lim,upper_lim,lower_lim,upper_lim], origin="lower")

# plot the contours
for plot_num in range(0,len(this_cube)):
    this_plot = this_cube[plot_num,:,:].astype('float')
    plt.contour(this_plot, levels=[1], colors="k", extent=[lower_lim,upper_lim,lower_lim,upper_lim], origin="lower")

ax.set_aspect('equal', 'box')
#plt.title("mask_406x406_rings_4quad_fits_coarse_20200409.fits")

asec_lim = np.multiply(np.shape(median_img)[0],0.0107)
asec_axis = np.arange(0,asec_lim,1)

# set width of tick marks to 1.0 asec
ax = plt.gca()
#ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

plt.xlabel("y (arcsec)", fontsize=18)
plt.ylabel("x (arcsec)", fontsize=18)

plt.xticks(rotation = 45, fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
#plt.show()
plt.savefig("junk.pdf")
