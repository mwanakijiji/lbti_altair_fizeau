#!/usr/bin/env python
# coding: utf-8

# This plots seeing values from HOSTS observations, obtained from https://lbti.ipac.caltech.edu/
# The purpose is to see under what conditions the phase loop has been able to close.

# Created 2020 July 10 by E.S.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# read in HOSTS data
df_hosts = pd.read_csv("data/lbti_l2ob_1787.csv")

# read in Altair observation data
df_altair_pretrim = pd.read_csv('fizeau_altair_180507_metadata.csv')
# only use data from the Altair observation
df_altair = df_altair_pretrim.where(np.logical_and(df_altair_pretrim["FRAMENUM"]>4404,
                                                   df_altair_pretrim["FRAMENUM"]<11334))
# make separate data for Altair when phase loop was open and when closed
#[print(i) for i in df_altair["PCCLOSED"]]
df_altair_phase_open = df_altair.where(df_altair["PCCLOSED"]!=1)
df_altair_phase_closed = df_altair.where(df_altair["PCCLOSED"]==1)

bin_width = 0.03
n_bins_hosts = int(np.divide(np.nanmax(df_hosts["Seeing(arcsec)"])-np.nanmin(df_hosts["Seeing(arcsec)"]),bin_width))
n_bins_altair_all = int(np.divide(np.nanmax(df_altair["SEEING"])-np.nanmin(df_altair["SEEING"]),bin_width))
n_bins_altair_open = int(np.divide(np.nanmax(df_altair_phase_open["SEEING"])-np.nanmin(df_altair_phase_open["SEEING"]),bin_width))
n_bins_altair_closed = int(np.divide(np.nanmax(df_altair_phase_closed["SEEING"])-np.nanmin(df_altair_phase_closed["SEEING"]),bin_width))

plt.hist(df_hosts["Seeing(arcsec)"],bins=n_bins_hosts,label="HOSTS\n(phase closed)",color="gray",alpha=0.5,density=True)
#plt.hist(df_altair["SEEING"],bins=n_bins_altair_all,histtype="step",label="Altair\n(this paper)",density=True)
plt.hist(df_altair_phase_open["SEEING"],bins=n_bins_altair_open,histtype="step",linewidth=3,color="#4daf4a",label="This work\n(phase open)",density=True)
plt.hist(df_altair_phase_closed["SEEING"],bins=n_bins_altair_closed,histtype="step",linewidth=3,color="#a65628",label="This work\n(phase closed)",density=True)
plt.ylabel("Normalized Histogram", fontsize=18)
plt.xlabel("Seeing (arcsec)", fontsize=18)
plt.xlim([0.25,2.25])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig("junk_seeing_hist.pdf")
#plt.show()

'''
plt.hist(df_altair["SMTTAU"],bins=80) # don't know if scaling is right
plt.hist(df["PWV"],bins=80)
plt.show()
'''
