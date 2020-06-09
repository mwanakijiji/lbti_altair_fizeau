# This reads in Jordan Stone's LEECH contrast curve for Altair for comparison

import pandas as pd
import matplotlib.pyplot as plt

# LEECH data
contrast_leech = pd.read_csv("data/HIP97649_LEECH.txt")
radii_leech = pd.read_csv("data/radii_LEECH.txt")

# our Fizeau data
data_fizeau =  pd.read_csv("data/modern_contrast_curve.csv")

plt.clf()
plt.plot(radii_leech,contrast_leech,
         label="LEECH\n(one telescope,\nt_int = 2873 sec.,\n83 deg rot,\nseeing 0.9)") # info from Table 4 in Stone+ 2018
plt.annotate("LEECH", xy=(1.5,14), rotation=-30, xycoords="data")
plt.plot(data_fizeau["rad_asec"],data_fizeau["del_m_modern"],
         label="This paper (Fizeau)\n(\nt_int = TBD sec.,\n83 deg rot,\nseeing ~0.9-1.4)")
plt.xlabel("Radius (asec)", fontsize=18)
plt.ylabel("$\Delta m$", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.legend(loc="lower center")
plt.gca().invert_yaxis()
plt.show()
#plt.savefig("junk_cc_comparison.pdf")
