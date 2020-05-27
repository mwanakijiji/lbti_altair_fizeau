#!/usr/bin/env python
# coding: utf-8

# This is for testing MTF quality by 
# 1. taking a cut through the middle
# 2. integrating portions of the MTF

# parent created 2019 June 23 by E.S.

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import gc
from astropy.io import fits
import pandas as pd
import glob
from mpl_toolkits.axes_grid1 import host_subplot
from collections import OrderedDict
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


dist_altair = 5.13 # pc

# significant distances at 'dist' pc
def au_2_freq_invarcsec(n_au,dist):
    l = dist/(n_au)
    return l

# ... and the inverse
def freq_invarcsec_2_au(freq,dist):
    n_au = np.divide(dist,freq)
    return n_au

# set x-axes
freq_x_axis_preshift = np.fft.fftfreq(360) # frequency axes (in pix^-1)
freq_x_axis = np.fft.fftshift(freq_x_axis_preshift)
# convert to mas^-1
freq_x_axis_invmas = np.divide(freq_x_axis,10.7)
freq_x_axis_invarcsec = (1e3)*freq_x_axis_invmas
# converted into aperture baselines in m
lambda_m = 4.05E-6
masec_per_rad = 360.*3600*1000/(2*np.pi)
arcsec_per_rad = 360.*3600/(2*np.pi)
baseline_x_axis_m = np.multiply(lambda_m*arcsec_per_rad,freq_x_axis_invarcsec) 

# also make it into a function, for making tick marks
def invmas_2_m(x_axis_invmas):
    axis_m = np.multiply(lambda_m*masec_per_rad,x_axis_invmas)
    return axis_m
def m_2_invmas(x_axis_m):
    axis_m = np.divide(x_axis_m,lambda_m)*np.divide(1.,masec_per_rad)
    return axis_m
def invarcsec_2_m(x_axis_invarcsec):
    axis_m = np.multiply(lambda_m*asec_per_rad,x_axis_invarcsec)
    return axis_m
def m_2_invarcsec(x_axis_m):
    axis_m = np.divide(x_axis_m,lambda_m)*np.divide(1.,arcsec_per_rad)
    return axis_m


# # Read in csvs of MTF data and plot them

# read in mtf data file

file_list = glob.glob("data/mtf_data_simulated_br_alpha.csv")
mtf_example = pd.read_csv(file_list[0])

# convert inverse mas to inverse as
freq_x_axis_invarcsec = (1e3)*freq_x_axis_invmas

plt.clf()

fig = plt.figure(figsize=(12,8))
host = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(bottom=0.2)

par2 = host.twiny()

offset = -50
new_fixed_axis = par2.get_grid_helper().new_fixed_axis

host.set_ylabel("MTF")
host.set_xlabel("Frequency (arcsec$^{-1}$)")

freq_habitable_inner = au_2_freq_invarcsec(2.5,dist=dist_altair)
freq_habitable_outer = au_2_freq_invarcsec(5.3,dist=dist_altair)
print("freq_habitable_outer")
print(freq_habitable_outer)
print("freq_habitable_inner")
print(freq_habitable_inner)

# location of max power of high-freq lobe
upper_lobe_df = pd.DataFrame.copy(mtf_example["med_strip"],deep=True)
upper_lobe_df[:220] = 0 # remove power from lower freqs
upper_lobe_df[300:] = 0 # remove power from numerical weirdness on edge
upper_lobe_df.idxmax()
res_upper_lobe_max = freq_x_axis_invarcsec[upper_lobe_df.idxmax()]
print("Frequency at max upper lobe power:")
print(res_upper_lobe_max)
print("... which corresponds to these AU:")
print(freq_invarcsec_2_au(res_upper_lobe_max,dist=dist_altair))

# power at 2.5 AU
host.axvline(x=freq_habitable_inner, linestyle="--", color="gray")
# power at 5.3 AU
host.axvline(x=freq_habitable_outer, linestyle="--", color="gray")
# power at 0.5 AU
host.axvline(x=au_2_freq_invarcsec(0.5,dist=dist_altair), linestyle="--", color="gray")
host.annotate("0.5 AU", (au_2_freq_invarcsec(0.5,dist=dist_altair)+0.2,0.8), rotation=0, fontsize=17)
# power at max
host.axvline(x=res_upper_lobe_max, linestyle="--", color="gray")
host.annotate("0.3 AU", (res_upper_lobe_max+0.2,0.8), rotation=0, fontsize=17)

# shade the area corresponding to radii of the habitable zone
host.axvspan(freq_habitable_outer, freq_habitable_inner, alpha=0.5, color='green')

p1, = host.plot(freq_x_axis_invarcsec,
             np.divide(mtf_example["med_strip"],np.max(mtf_example["med_strip"])), 
                    color="r", linewidth=0.5, alpha=0.7, label="Br-alpha")

# the perfect MTF
mtf_br_alpha_perfect = pd.read_csv("data/mtf_data_simulated_br_alpha.csv")
p1, = host.plot(freq_x_axis_invarcsec,
         np.divide(mtf_br_alpha_perfect["med_strip"],np.max(mtf_br_alpha_perfect["med_strip"])), 
                color="k", label="Modeled")

# add secondary axis across the top
def freq2baseline(x):
    return x * arcsec_per_rad * lambda_m
def baseline2freq(x):
    return x / (arcsec_per_rad * lambda_m)
secax = host.secondary_xaxis('top', functions=(freq2baseline, baseline2freq))
secax.set_xlabel('Aperture baseline (m)')

host.set_xlim(0,1e3*m_2_invmas(22.3))
#par2.set_xlim(0,22.3)
par2.set_ylim(0, 1.1)

#host.axis["bottom"].label.set_color(p1.get_color())
#par2.axis["bottom"].label.set_color(p3.get_color())

params = {'axes.labelsize': 22,
          'axes.titlesize':20, 
          'legend.fontsize': 20, 
          'xtick.labelsize': 20, 
          'ytick.labelsize': 20}

matplotlib.rcParams.update(params)

plt.show()
#plt.savefig("test.pdf")
