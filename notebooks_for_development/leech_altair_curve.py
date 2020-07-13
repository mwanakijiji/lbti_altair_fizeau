# This reads in Jordan Stone's LEECH contrast curve for Altair for comparison

import pandas as pd
import matplotlib.pyplot as plt

# LEECH data
contrast_leech = pd.read_csv("data/HIP97649_LEECH.txt")
radii_leech = pd.read_csv("data/radii_LEECH.txt")

# our Fizeau data
#data_fizeau =  pd.read_csv("data/modern_contrast_curve.csv")

# (ersatz Fizeau data)
data_fizeau =  pd.read_csv("data/modern_curve_20200713.csv")
print(data_fizeau.keys())
#data_fizeau["del_m_modern"] = data_fizeau["del_m_5_sig"]

plt.clf()
plt.annotate("LEECH", xy=(1.5,14.5), rotation=-30, xycoords="data", fontsize=14)
plt.annotate("This paper\n(Fizeau)", xy=(2.1,9.5), rotation=0, xycoords="data", fontsize=14)
plt.axvline(x=1.,ymin=0.6,linestyle=":",linewidth=4,color="k",alpha=0.5) # radius beyond which there is detector overshoot to southeast
plt.annotate("detector\novershoot", xy=(1.05,8.1), rotation=90, xycoords="data", fontsize=11)
plt.arrow(x=1., y=8.5, dx=0.1, dy=0, head_length=0.03, head_width=0.2,
        linewidth=3, fc="k", alpha=1)
plt.axvline(x=1.4,ymin=0.6,linestyle="--",linewidth=4,color="k",alpha=0.5) # approx. control radius
plt.annotate("$r_{c}$", xy=(1.45,8.3), xycoords="data", fontsize=11)
plt.plot(radii_leech,contrast_leech,
         label="LEECH\n(one telescope,\nt_int = 2873 sec.,\n83 deg rot,\nseeing 0.9)",
         linewidth=4,color="teal") # info from Table 4 in Stone+ 2018
plt.plot(data_fizeau["rad_asec"],data_fizeau["del_m_modern"],
         label="This paper (Fizeau)\n(\nt_int = TBD sec.,\n83 deg rot,\nseeing ~0.9-1.4)",
         linewidth=4,color="indianred")
plt.xlabel("Radius (asec)", fontsize=18)
plt.ylabel("$\Delta m$", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim([5,16])
#plt.legend(loc="lower center")
plt.gca().invert_yaxis()
plt.tight_layout()
#plt.show()
plt.savefig("junk_cc_comparison.pdf")
