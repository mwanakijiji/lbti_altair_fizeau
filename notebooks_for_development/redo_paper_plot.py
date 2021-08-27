#!/usr/bin/env python
# coding: utf-8

# This makes another plot of the Fig. 11 in the Altair paper, which was
# criticized by a reviewer

# Created 2021 Aug. 12 by E.S.

import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np

stem = "./data/redo_paper_plot"

hdul_w_planet = fits.open(stem + "/w_planet.fits")
hdul_wo_planet = fits.open(stem + "/wo_planet.fits")

resids = np.subtract(hdul_w_planet[0].data,hdul_wo_planet[0].data)

low_pass = 203-40
high_pass = 203+40
'''
# no planet
plt.imshow(hdul_wo_planet[0].data[low_pass:high_pass,low_pass:high_pass], origin="lower")
plt.show()

# w planet
plt.imshow(hdul_w_planet[0].data[low_pass:high_pass,low_pass:high_pass], origin="lower")
plt.show()

# resids
plt.imshow(resids[low_pass:high_pass,low_pass:high_pass], origin="lower")
plt.show()
'''

# all together now
plt.clf()
fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True, constrained_layout=True)
im0 = axs[0].imshow(hdul_wo_planet[0].data[low_pass:high_pass,low_pass:high_pass],
              origin="upper",extent=[-0.428,0.428,0.428,-0.428], cmap="gray", vmin=-483, vmax=556) # vmin=-483, vmax=556
#im0 = axs[0].imshow(hdul_wo_planet[0].data[low_pass:high_pass,low_pass:high_pass],
#              origin="upper",extent=[-0.428,-0.428,0.428,0.428], cmap="gray", vmin=-483, vmax=556)

im1 = axs[1].imshow(hdul_w_planet[0].data[low_pass:high_pass,low_pass:high_pass],
              origin="upper",extent=[-0.428,0.428,0.428,-0.428], cmap="gray", vmin=-483, vmax=556)
im2 = axs[2].imshow(resids[low_pass:high_pass,low_pass:high_pass],
              origin="upper",extent=[-0.428,0.428,0.428,-0.428], cmap="gray", vmin=-11, vmax=87)
axs[1].invert_yaxis()

axs[0].set_ylabel("Arcsec", fontsize=17)

axs[0].set_title("No companion", fontsize=20)
axs[1].set_title("Companion", fontsize=20)
axs[2].set_title("Residuals", fontsize=20)

axs[0].tick_params(axis='y', labelsize=13)
axs[0].tick_params(axis='x', labelsize=13)
axs[1].tick_params(axis='x', labelsize=13)
axs[2].tick_params(axis='x', labelsize=13)

axs[0].set_aspect('equal')
axs[1].set_aspect('equal')
axs[2].set_aspect('equal')

# colorbar 1
cbaxes1 = fig.add_axes([0.1, 0.0, 0.525, 0.07]) # left, bottom, width, height
cb1 = plt.colorbar(im0, ax=axs[:2], cax = cbaxes1, orientation="horizontal")

# colorbar 2
cbaxes2 = fig.add_axes([0.65, 0.0, 0.25, 0.07]) # left, bottom, width, height
cb2 = plt.colorbar(im2, ax=axs[2], cax = cbaxes2, orientation="horizontal")

cb1.ax.tick_params(labelsize=13)
cb2.ax.tick_params(labelsize=13)

# adjust size of subplots
plt.subplots_adjust(left=0.1,
                    bottom=0.2,
                    right=0.9,
                    top=0.9,
                    wspace=0.1,
                    hspace=0.6)

# add compass
axs[0].annotate("",
            xy=(0.35, 0.0), xycoords='data',
            xytext=(0.35, -0.31), textcoords='data',
            arrowprops=dict(arrowstyle="->",lw=3,
                            connectionstyle="arc3")
            )
axs[0].annotate("",
            xy=(0.05, -0.3), xycoords='data',
            xytext=(0.36, -0.3), textcoords='data',
            arrowprops=dict(arrowstyle="->",lw=3,
                            connectionstyle="arc3")
            )
axs[0].annotate("N",fontsize=20,
            xy=(0.05, -0.3), xycoords='data',
            xytext=(0.3, 0.0), textcoords='data'
            )
axs[0].annotate("E",fontsize=20,
            xy=(0.05, -0.3), xycoords='data',
            xytext=(-0.02, -0.34), textcoords='data'
            )

#plt.tight_layout()
plt.savefig("junk.png", dpi=400, bbox_inches="tight")
