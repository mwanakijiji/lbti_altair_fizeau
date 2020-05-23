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
# be sure to comment/uncomment the plot commands with the 'extent' arguments

fig3 = plt.figure(constrained_layout=True)
gs = fig3.add_gridspec(ncols=1, nrows=5)

f3_ax1 = fig3.add_subplot(gs[0:4,:])

#f, ax = plt.subplots(2, 1, sharex=True)
#f, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})
#a0.plot(x, y)
#a1.plot(y, x)

# lambda/D
test_data_frame, hdr = fits.getdata("data/mask_406x406_rings_4quad_fits_coarse_20200409.fits", 0, header=True)
lower_lim_x = -2.1721
upper_lim_x = 2.1721
lower_lim_y = -2.1721
upper_lim_y = 2.1721

## lambda/D
# plot the grey (not considered) regions
this_cube = np.copy(test_data_frame)
median_img = np.nanmedian(this_cube,axis=0)
f3_ax1.imshow(median_img, cmap="gray", vmin=-1, vmax=0.2,
           extent=[lower_lim_x,upper_lim_x,lower_lim_y,upper_lim_y], origin="lower")
# plot the white (considered) regions
for plot_num in range(0,len(this_cube)):
    this_plot = this_cube[plot_num,:,:].astype('float')
    this_plot[this_plot == 0] = np.nan # this to a
    f3_ax1.imshow(this_plot, cmap="gray", vmin=-3, vmax=1,
               extent=[lower_lim_x,upper_lim_x,lower_lim_y,upper_lim_y], origin="lower")
    f3_ax1.contour(this_plot, levels=[1], colors="k", extent=[lower_lim_x,upper_lim_x,lower_lim_y,upper_lim_y], origin="lower")
# plot the contours
for plot_num in range(0,len(this_cube)):
    this_plot = this_cube[plot_num,:,:].astype('float')
    f3_ax1.contour(this_plot, levels=[1], colors="k", extent=[lower_lim_x,upper_lim_x,lower_lim_y,upper_lim_y], origin="lower")
#ax[0].set_aspect('equal', 'box')
f3_ax1.axes.get_xaxis().set_ticklabels([])

# lambda/B (note we are redefining variables!)
f3_ax2 = fig3.add_subplot(gs[4,:])

test_data_frame, hdr = fits.getdata("data/mask_406x406_center_strip_width_2_FWHM_lamb_over_B.fits", 0, header=True)
test_data_frame = test_data_frame[:,156:250,:]
lower_lim_x_lambda_B = -2.1721
upper_lim_x_lambda_B = 2.1721
lower_lim_y_lambda_B = -0.5
upper_lim_y_lambda_B = 0.5
this_cube = np.copy(test_data_frame)

## lambda/B_CC
# plot the grey (not considered) regions
aspect_const = 0.82 # 0.837
median_img = np.nanmedian(this_cube,axis=0)
f3_ax2.imshow(median_img, cmap="gray", vmin=-1, vmax=0.2,
           extent=[lower_lim_x_lambda_B,upper_lim_x_lambda_B,lower_lim_y_lambda_B,upper_lim_y_lambda_B],
           origin="lower", aspect=aspect_const)
# plot the white (considered) regions
for plot_num in range(0,len(this_cube)):
    this_plot = this_cube[plot_num,:,:].astype('float')
    this_plot[this_plot == 0] = np.nan # this to a
    f3_ax2.imshow(this_plot, cmap="gray", vmin=-3, vmax=1,
               extent=[lower_lim_x_lambda_B,upper_lim_x_lambda_B,lower_lim_y_lambda_B,upper_lim_y_lambda_B],
               origin="lower", aspect=aspect_const)
    f3_ax2.contour(this_plot, levels=[1], colors="k",
                extent=[lower_lim_x_lambda_B,upper_lim_x_lambda_B,lower_lim_y_lambda_B,upper_lim_y_lambda_B],
                origin="lower", aspect=aspect_const)
# plot the contours
for plot_num in range(0,len(this_cube)):
    this_plot = this_cube[plot_num,:,:].astype('float')
    f3_ax2.contour(this_plot, levels=[1], colors="k",
                extent=[lower_lim_x_lambda_B,upper_lim_x_lambda_B,lower_lim_y_lambda_B,upper_lim_y_lambda_B],
                origin="lower", aspect=aspect_const)
#ax[1].set_aspect('equal', 'box')

#asec_lim = np.multiply(np.shape(median_img)[0],0.0107)
#asec_axis = np.arange(0,asec_lim,1)

# set width of tick marks to 1.0 asec
'''
ax = plt.gca()
tick_spacing = 0.5
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
'''
#ax[1].ylabel("y (arcsec)", fontsize=18)
f3_ax1.set_ylabel("y (arcsec)", fontsize=18)
#f3_ax2.set_ylabel("y (arcsec)", fontsize=18)
f3_ax2.set_xlabel("x (arcsec)", fontsize=18)
#ax[1].set(xlabel='x-label', ylabel='y-label')

#ax[1].set_xticks(rotation = 45, fontsize=14)
#ax[0].set_yticks(fontsize=14)
#ax[1].set_yticks(fontsize=14)

#plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=45)
plt.tight_layout()
#plt.show()
plt.savefig("junk.pdf")
