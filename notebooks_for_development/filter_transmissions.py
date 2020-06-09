# Plots filter transmissions

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Transform
from matplotlib.ticker import (
    AutoLocator, AutoMinorLocator, FuncFormatter)
from scipy import interpolate

# read in curves
naco_trans = pd.read_csv("data/Paranal_NACO.NB405.dat.txt", names = ["angstrom", "transm"], delim_whitespace=True)
lmir_bralpha_trans = pd.read_csv("data/br-alpha_NDC.txt", delim_whitespace=True)
lmir_bralpha_trans["Wavelength_angstr"] = np.multiply(10000.,lmir_bralpha_trans["Wavelength"])

# plot filter curves
plt.clf()
plt.plot(np.divide(naco_trans["angstrom"],1e4), naco_trans["transm"], label = "NACO NB4.05", linewidth=4)
plt.plot(np.divide(lmir_bralpha_trans["Wavelength_angstr"],1e4), lmir_bralpha_trans["Trans_77"],
         label = "LMIRcam Br-"+r"$\alpha$", linewidth=4)
plt.xlim([3.9750,4.1550])
plt.ylim([0,1])
plt.xlabel("Wavelength ("+r"$\mu$m"+")", fontsize=18)
plt.ylabel("Transmission", fontsize=18)
#locs, labels = xticks()
plt.xticks(np.arange(3.98, 4.16, step=0.02), rotation=45, fontsize=14)
plt.yticks(np.arange(0, 1.1, step=0.2), fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig("junk_filter_comparizon_v2.pdf")
