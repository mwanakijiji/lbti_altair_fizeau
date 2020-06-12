# This makes a plot of atmsopheric transmission over the bandpass

# Parent notebook created 2020 Apr. 30 by E.S.

# For the math, see research notebook fizeau_altair.tex on date 2020 Apr. 13
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import os
import pysynphot as S

# Naco NB405 filter
nb405_transmission = pd.read_csv("data/Paranal_NACO.NB405.dat.txt",
                                 names=["wavel_angs", "transmission"], delim_whitespace=True)

# ### Read in atmospheric transmission
# source: https://atran.arc.nasa.gov/cgi-bin/atran/atran.cgi
# INPUT PARAMS for atran.plt.12172.dat:
'''
Altitude    :    10567.0000
 Water Vapor :    11.0000000
 Num layers  :            2
 Zenith Angle:    30.0000000
 Obs Lat     :    30.0000000
 Minimum Wave:    3.98000002
 Maximum Wave:    4.13000011
'''
trans_df = pd.read_csv("data/atran.plt.12172.dat", usecols=[1,2],
                       names=["wavel_um","transmission"], delim_whitespace=True)
# add column of wavelength in angstroms
trans_df["wavel_angs"] = (1e4)*trans_df["wavel_um"]

# make smoothed version of the atmospheric transmission before interpolating
smoothed_atm_trans = scipy.signal.medfilt(trans_df["transmission"],kernel_size=401)
atm_transmission_filter = np.interp(nb405_transmission["wavel_angs"].values,
                                    trans_df["wavel_angs"],
                                    smoothed_atm_trans)

plt.clf()
#color = 'tab:red'
plt.xlabel('Wavelength ($\AA$)', fontsize=18)
plt.ylabel('Transmission', fontsize=18)
plt.plot(nb405_transmission["wavel_angs"],nb405_transmission["transmission"],
         label="NACO NB4.05 filter ($R_{\lambda}$)", linewidth=4)
plt.plot(trans_df["wavel_angs"],trans_df["transmission"],
         label="Atmosphere (high-res)", linewidth=1)
plt.plot(nb405_transmission["wavel_angs"].values, atm_transmission_filter,
         label="Atmosphere ($T_{\lambda}$)", linewidth=4)
plt.xlim([39850,41250])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc="upper right")
plt.show()
