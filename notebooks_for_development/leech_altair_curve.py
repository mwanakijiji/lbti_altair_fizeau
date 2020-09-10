# This reads in Jordan Stone's LEECH contrast curve for Altair for comparison

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# LEECH data
contrast_leech = pd.read_csv("data/HIP97649_LEECH.txt")
radii_leech = pd.read_csv("data/radii_LEECH.txt")

# our Fizeau data
data_fizeau =  pd.read_csv("data/modern_curve_20200713.csv")
print(data_fizeau.keys())
#data_fizeau["del_m_modern"] = data_fizeau["del_m_5_sig"]

## offsets 'alpha' for rescaling (see notebook contrast_curve_rescalings.ipynb)
# 10x flux change from removal of ND filter; note the '+' for treating it as an offset
alpha_flux = +2.5*np.log10(np.sqrt(10))
# collecting area change if 2->1 aperture
alpha_area = +2.5*np.log10(np.sqrt(0.5))
# number of nights
t_int_leech = 2873 # LEECH integration time (sec; see Table 4 in Stone+ 2018)
t_int_fiz = 445 # Fizeau integration time (sec; and only valid <1 asec)
N_night = np.divide(t_int_leech,t_int_fiz)
alpha_nights = +2.5*np.log10(np.sqrt(N_night))
# rotation
rot_leech = 83 # LEECH rotation (degrees)
rot_fiz = 32.6 # Fizeau rotation (degrees)
alpha_rotation = +2.5*np.log10(np.sqrt(np.divide(rot_leech,rot_fiz)))
# all together
alpha_everything = +2.5*np.log10(np.sqrt(10*0.5*N_night*np.divide(rot_leech,rot_fiz)))

# now rescale the Fizeau curve (only use within rho=1.05 asec)
flux_rescaled_fizeau = np.add(alpha_flux,data_fizeau["del_m_modern"].where(data_fizeau["rad_asec"] < 1.05))
area_rescaled_fizeau = np.add(alpha_area,data_fizeau["del_m_modern"].where(data_fizeau["rad_asec"] < 1.05))
nights_rescaled_fizeau = np.add(alpha_nights,data_fizeau["del_m_modern"].where(data_fizeau["rad_asec"] < 1.05))
rotation_rescaled_fizeau = np.add(alpha_rotation,data_fizeau["del_m_modern"].where(data_fizeau["rad_asec"] < 1.05))
all_together_rescaled_fizeau = np.add(alpha_everything,data_fizeau["del_m_modern"].where(data_fizeau["rad_asec"] < 1.05))

print("Flux rescaling (mag):")
print(alpha_flux)
print("Collecting area rescaling (mag):")
print(alpha_area)
print("Equivalent nights rescaling (mag):")
print(alpha_nights)
print("Rotation rescaling (mag):")
print(alpha_rotation)
print("Everything rescaling (mag):")
print(alpha_everything)

plt.clf()
plt.annotate("LEECH", xy=(1.5,14.5), rotation=-30, xycoords="data", fontsize=14)
plt.annotate("This paper\n(Fizeau)", xy=(2.1,9.5), rotation=0, xycoords="data", fontsize=14)
plt.axvline(x=1.,ymin=0.6,linestyle=":",linewidth=4,color="k",alpha=0.5) # radius beyond which there is detector overshoot to southeast
plt.annotate("detector\novershoot", xy=(1.05,8.1), rotation=90, xycoords="data", fontsize=11)
plt.arrow(x=1., y=8.5, dx=0.1, dy=0, head_length=0.03, head_width=0.2,
        linewidth=3, fc="k", alpha=1)
plt.axvline(x=1.4,ymin=0.6,linestyle="--",linewidth=4,color="k",alpha=0.5) # approx. control radius
plt.annotate("$r_{c}$", xy=(1.45,8.3), xycoords="data", fontsize=11)
plt.plot(radii_leech,contrast_leech,linewidth=4,color="teal") # info from Table 4 in Stone+ 2018
plt.plot(data_fizeau["rad_asec"],data_fizeau["del_m_modern"],linewidth=4,color="indianred")

# offsets
plt.plot(data_fizeau["rad_asec"].where(data_fizeau["rad_asec"] < 1.05),flux_rescaled_fizeau,
        label="No ND filter",linewidth=4,linestyle="--")
plt.plot(data_fizeau["rad_asec"].where(data_fizeau["rad_asec"] < 1.05),area_rescaled_fizeau,
        label="Collecting area",linewidth=4,linestyle="--")
plt.plot(data_fizeau["rad_asec"].where(data_fizeau["rad_asec"] < 1.05),nights_rescaled_fizeau,
        label="Nights (1 to 6.46)",linewidth=4,linestyle="--")
plt.plot(data_fizeau["rad_asec"].where(data_fizeau["rad_asec"] < 1.05),rotation_rescaled_fizeau,
        label="Rotation",linewidth=4,linestyle="--")
plt.plot(data_fizeau["rad_asec"].where(data_fizeau["rad_asec"] < 1.05),all_together_rescaled_fizeau,
        label="All",linewidth=4,linestyle="--")

plt.legend(bbox_to_anchor=(0.65, 0.22),loc=(2,11))
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
