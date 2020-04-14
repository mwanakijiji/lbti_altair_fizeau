import numpy as np
import pandas as pd
from modules import *
import matplotlib.pyplot as plt
from matplotlib.transforms import Transform
from matplotlib.ticker import (
    AutoLocator, AutoMinorLocator, FuncFormatter)
from scipy import interpolate


# MOVE THIS TO INIT
def asec_to_AU(input_asec):

    # convert asec to AU
    dist_altair = 5.130 # pc
    output_AU = np.multiply(dist_altair,input_asec)

    return output_AU


def AU_to_asec(input_AU):

    # convert AU to asec
    dist_altair = 5.130 # pc
    output_asec = np.divide(input_AU,dist_altair)

    return output_asec

# absolute magnitude of host star in NB205 filter
## ## PLACEHOLDER VALUE HERE; CURRENT VALUE FROM NOTEBOOK
## ## determine_abs_mag_altair.ipynb; NOT DOUBLE-CHECKED;
## ## REPLACE LATER (NOTE ALSO THAT ALTAIR IS A VARIABLE SOURCE!)
abs_mag_altair_nb405 = 1.87

# Obtain input data
# pipeline will output a fake companion amplitude which is normalized, linearly, to the host star (amplitude of 1)
# make/read in a contrast curve, where contrast is defined as the flux ratio F_planet/F_star where detection
# has 5-sigma significance
## contrast_df = pd.read_csv("data/fake_contrast_curve.csv")
contrast_df = pd.read_csv("./notebooks_for_development/data/classical_curve_20200316.csv")


# In[5]:


# convert linear empirical contrast to del_mag
contrast_df["del_mag_LMIR"] = -2.5*np.log10(contrast_df["contrast_lim"])

# convert del_mag (between planet and host star) to abs. mag (of planet)
contrast_df["abs_mag_LMIR"] = np.add(contrast_df["del_mag_LMIR"],abs_mag_altair_nb405)

# convert asec to AU
dist_altair = 5.130 # pc
contrast_df["AU"] = np.multiply(dist_altair,contrast_df["asec"])


# In[6]:


# read in models

# these are from
# AMES-Cond: https://phoenix.ens-lyon.fr/Grids/AMES-Cond/ISOCHRONES/model.AMES-Cond-2000.M-0.0.NaCo.Vega

# Br-alpha filter is model_data["NB4.05"], in Vega magnitudes

model_data = pd.read_csv("data/1gr_data.txt", delim_whitespace=True)


# In[7]:


# read in NACO transmission curve for comparison

naco_trans = pd.read_csv("data/Paranal_NACO.NB405.dat.txt", names = ["angstrom", "transm"], delim_whitespace=True)
lmir_bralpha_trans = pd.read_csv("data/br-alpha_NDC.txt", delim_whitespace=True)
lmir_bralpha_trans["Wavelength_angstr"] = np.multiply(10000.,lmir_bralpha_trans["Wavelength"])


# In[7]:


# plot filter curves

plt.clf()
plt.plot(naco_trans["angstrom"], naco_trans["transm"], label = "NACO NB4.05")
plt.plot(lmir_bralpha_trans["Wavelength_angstr"], lmir_bralpha_trans["Trans_77"],
         label = "LMIR Br-"+r"$\alpha$ (T = 77 K)")
plt.xlim([39750,41550])
plt.xlabel("Wavelength ("+r"$\AA$"+")")
plt.ylabel("Transmission")
plt.legend()
plt.show()


# In[17]:


model_data


# ### Interpolate the models to map absolute mag to mass

# In[8]:


# make function to interpolate models

f_abs_mag_2_mass = interpolate.interp1d(model_data["NB4.05"],model_data["M/Ms"])

# ... and its inverse

f_mass_2_abs_mag = interpolate.interp1d(model_data["M/Ms"],model_data["NB4.05"])


# In[57]:


# plot model data and interpolation

plt.clf()
plt.plot(model_data["NB4.05"], model_data["M/Ms"], color="blue", label="model points", marker="o")
plt.scatter(contrast_df["abs_mag_LMIR"], contrast_df["masses_LMIR"], color="orange",
            label="contrast curve interpolation")
plt.xlabel("abs_mag LMIR")
plt.ylabel("M/M_solar")
plt.legend()
plt.show()


# In[13]:


# return masses (M/M_solar) corresponding to our contrast curve

contrast_df["masses_LMIR"] = f_abs_mag_2_mass(contrast_df["del_mag_widthFWHM"])


# return more masses corresponding to interpolations at intervals

mass_intervals = [0.5,0.6,0.7,0.8,0.9,1.0]
annotate_mass_intervals = ["0.5 Ms","0.6 Ms","0.7 Ms","0.8 Ms","1.0 Ms","1.0 Ms"]
abs_mag_intervals = f_mass_2_abs_mag(mass_intervals)


# ### Make plot
#
# #### left y-axis: abs mag
# #### bottom x-axis: asec
# #### right y-axis: M/Ms
# #### top x-axis: AU

# In[17]:


#f = lambda q: q
#finv = lambda x: np.log10(2+x)+np.cos(x)

fig, ax = plt.subplots()
fig.suptitle("Contrast curve\n(based on M_altair = 1.8; NOT QUADRUPLE-CHECKED")
ax2 = ax.twinx()
ax.set_xlim([0,2.2]) # 0 to 2.2 asec
ax.get_shared_y_axes().join(ax,ax2)

ax.set_ylabel('Abs mag (LMIR)')
ax.set_xlabel('Angle (asec)')

ax.plot(contrast_df["asec"], contrast_df["abs_mag_LMIR"])

# secondary x axis on top
secax_x = ax.secondary_xaxis('top', functions=(asec_to_AU, AU_to_asec))
secax_x.set_xlabel('Distance (AU)')

# draw horizontal lines corresponding to certain masses
for t in range(0,len(mass_intervals)):
    ax.axhline(y=abs_mag_intervals[t], linestyle="--", color="k")
    ax.annotate(annotate_mass_intervals[t],
                xy=(0.4,abs_mag_intervals[t]),
                xytext=(0,0), textcoords="offset points")

ax2.yaxis.set_major_formatter(FuncFormatter(lambda t,pos: f"{f_abs_mag_2_mass(t):.2f}"))
ax2.set_ylabel('Masses (M/Ms)')
plt.gca().invert_yaxis()
plt.show()


# In[11]:


contrast_df.keys()


# In[ ]:


# sources of error:
# 1. uncertainty of distance from parallax
# 2. wavelength dependency of atmospheric transmission -> absolute magnitude of planet
# 3. small differences in filter bandpass between LMIR, NB4.05



END PASTED IN
##############

def blahblah():
    '''
    Make a circular mask somewhere in the input image
    returns 1=good, nan=bad/masked

    INPUTS:
    input_array: the array to mask
    mask_center: the center of the mask, in (y,x) input_array coords
    mask_radius: radius of the mask, in pixels
    invert: if False, area INSIDE mask region is masked; if True, area OUTSIDE

    OUTPUTS:
    mask_array: boolean array (1 and nan) of the same size as the input image
    '''

    return


class OneDimContrastCurve:
    '''
    Produces a 1D contrast curve (for regime of large radii)
    '''

    def __init__(self,
                 config_data = config):
        '''
        INPUTS:
        config_data: configuration data, as usual
        '''

        self.config_data = config_data


        ##########


    def __call__(self,
                 csv_file):
        '''
        Read in the csv with detection information and make a 1D contrast curve

        INPUTS:

        csv_file: absolute name of the file which contains the detection information for all fake planet parameters
        '''

        # CODE HERE




def main():
    '''
    Take an input 1D contrast curve and convert it to masses
    '''

    # configuration data
    config = configparser.ConfigParser() # for parsing values in .init file
    config.read("/modules/config.ini")

    # make a 1D contrast curve (PLACEHOLDER)
    one_d_contrast = OneDimContrastCurve(csv_file = config["data_dirs"]["DIR_S2N"] + \
        config["file_names"]["DETECTION_CSV"])
    one_d_contrast()

    '''
    # make a 2D sensitivity map
    two_d_sensitivity = TwoDimSensitivityMap(csv_file = config["data_dirs"]["DIR_S2N"] + config["file_names"]["DETECTION_CSV"])
    two_d_sensitivity()
    '''
