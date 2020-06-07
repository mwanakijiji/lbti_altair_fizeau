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

# read in the five baselines
test_frame_deg089pt96, hdr = fits.getdata("data/deg089pt96.fits", 0, header=True)
test_frame_deg096pt63, hdr = fits.getdata("data/deg096pt63.fits", 0, header=True)
test_frame_deg103pt43, hdr = fits.getdata("data/deg103pt43.fits", 0, header=True)
test_frame_deg109pt218, hdr = fits.getdata("data/deg109pt218.fits", 0, header=True)
test_frame_deg129pt68, hdr = fits.getdata("data/deg129pt68.fits", 0, header=True)

# sum them
test_frame_sum = np.sum([test_frame_deg089pt96,
                                test_frame_deg096pt63,
                                test_frame_deg103pt43,
                                test_frame_deg109pt218,
                                test_frame_deg129pt68], axis=0)

test_frame_sum[np.where(np.abs(test_frame_sum)>0.001)] = 1

import ipdb; ipdb.set_trace()
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
