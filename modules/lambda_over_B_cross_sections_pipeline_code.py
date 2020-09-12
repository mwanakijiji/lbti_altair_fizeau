# This takes host-star-subtracted frames and finds residuals, with a view to actually using
# this code in the pipeline

# created from parent 2020 May 16 by E.S.


import urllib
import numpy as np
import pandas as pd
import glob
import os
from scipy import ndimage, misc, stats
from astropy.io import fits
import matplotlib.pyplot as plt


stem = "/Users/nyumbani/Downloads/"


# In[3]:


def shave_and_rotate(img, angle):
    '''
    Shave off edges of frames (to get rid of the NaNs) and rotate
    '''

    edge_size_shave = 150 # shave off the edges

    # shave off edges symmetrically
    img_shaved = img[edge_size_shave:int(np.shape(img)[0]-edge_size_shave),
                     edge_size_shave:int(np.shape(img)[1]-edge_size_shave)]

    img_shaved_rotated = ndimage.rotate(img_shaved, angle, reshape=False)

    return img_shaved_rotated


# In[4]:


def pluck_interesting_file_name(file_names, comp_ampl_pass, dist_asec_pass):
    '''
    Function to loop through the arrays of file names and pluck out the file name of interest
    '''

    found_one_ticker = 0 # to indicate that the file name was found, and to avoid repeats

    for file_num in range(0,len(file_names)):

        # find current (i.e., not necessarily the starting) amplitude of the fake planet
        planet_current_ampl = float(file_names[file_num].split("_")[-2])

        # find planet location in asec from file name
        planet_loc_asec = float(file_names[file_num].split("_")[-3])

        # grab the interesting name
        if ((np.round(planet_current_ampl,2) == comp_ampl_pass) and (np.round(planet_loc_asec,2) == dist_asec_pass)):

            file_name_this_strip_of_interest = file_names[file_num]
            found_one_ticker += 1

    if (found_one_ticker > 1):

        print("Something is wrong-- found more than 1 matching name!")

    elif (found_one_ticker == 0):

        print("No matching files!")

    else:

        print("Found matching file \n" + str(file_name_this_strip_of_interest))

    return file_name_this_strip_of_interest


# In[24]:


def do_KS(empirical_sample_1,empirical_sample_2):
    '''
    2-sample KS test
    '''

    n1 = len(empirical_sample_1)
    n2 = len(empirical_sample_2)

    # Test the null hypothesis H0 which states 'the two samples are NOT from different parent populations'

    # The python function returns D, the max|distrib_1 - distrib_2|, so this needs to be multiplied by
    # scale_fac = (mn/(m+n))^(1/2)

    D, p_val = stats.ks_2samp(empirical_sample_1, empirical_sample_2)
    scale_fac = np.sqrt(np.divide(n1+n2,n1*n2))

    # critical value is (i.e., if D is above this value, we reject the null hypothesis)
    val_crit = np.multiply(1.358,scale_fac)

    # set significance level
    #alpha = 0.05

    return D, val_crit, p_val


# In[6]:


# retrieve ALL file names (these should be constant)

# glob of file names of ADI frames of A block strip 0 of 4
# (planets are in this strip)
file_names_strip_1_of_4_planetsInStrip0 = list(glob.glob(stem+"job_3203135/*.fits"))

# glob of file names of ADI frames of D block strip 1 of 4
# (planets present in strip 0 of 4)
file_names_strip_1_of_4_planetsInStrip0 = list(glob.glob(stem+"job_3216450/*.fits"))

# glob of file names of ADI frames of D block strip 2 of 4
# (planets present in strip 0 of 4)
file_names_strip_2_of_4_planetsInStrip0 = list(glob.glob(stem+"job_3216451/*.fits"))

# glob of file names of ADI frames of D block strip 3 of 4
# (planets present in strip 0 of 4)
file_names_strip_3_of_4_planetsInStrip0 = list(glob.glob(stem+"job_3216452/*.fits"))

# glob of file names of ADI frames of D block strip 4 of 4
# (planets present in strip 0 of 4)
file_names_strip_4_of_4_planetsInStrip0 = list(glob.glob(stem+"job_3216453/*.fits"))


# In[7]:


# for a single combination of companion distance and amplitude,
# read in arrays, process them, find residuals, and calculate KS test

############################
# BEGIN USER INPUTS 1 OF 2
############################

# choose the arrays; note that we do not say which strip has the planets until later
file_names_strip_0_of_4 = file_names_strip_1_of_4_planetsInStrip0
file_names_strip_1_of_4 = file_names_strip_1_of_4_planetsInStrip0
file_names_strip_2_of_4 = file_names_strip_2_of_4_planetsInStrip0
file_names_strip_3_of_4 = file_names_strip_3_of_4_planetsInStrip0
file_names_strip_4_of_4 = file_names_strip_4_of_4_planetsInStrip0

comp_ampl = 0.1
dist_asec = 0.52

new_filename = "test.png"

##########################
# END USER INPUTS 1 OF 2
##########################


# In[8]:


# pluck out the interesting file names

file_name_strip_0_of_4 = pluck_interesting_file_name(file_names_strip_0_of_4,
                                                     comp_ampl_pass=comp_ampl,dist_asec_pass=dist_asec)
file_name_strip_1_of_4 = pluck_interesting_file_name(file_names_strip_1_of_4,
                                                     comp_ampl_pass=comp_ampl,dist_asec_pass=dist_asec)
file_name_strip_2_of_4 = pluck_interesting_file_name(file_names_strip_2_of_4,
                                                     comp_ampl_pass=comp_ampl,dist_asec_pass=dist_asec)
file_name_strip_3_of_4 = pluck_interesting_file_name(file_names_strip_3_of_4,
                                                     comp_ampl_pass=comp_ampl,dist_asec_pass=dist_asec)
file_name_strip_4_of_4 = pluck_interesting_file_name(file_names_strip_4_of_4,
                                                     comp_ampl_pass=comp_ampl,dist_asec_pass=dist_asec)


# In[9]:


# read in and process the images
# (note we still don't define which strip has the planets)

image_stripe_0 = fits.getdata(file_name_strip_0_of_4,0,header=False)
img_processed_stripe_0 = shave_and_rotate(image_stripe_0,angle=39.68)

image_stripe_1 = fits.getdata(file_name_strip_1_of_4,0,header=False)
img_processed_stripe_1 = shave_and_rotate(image_stripe_1,angle=19.218)

image_stripe_2 = fits.getdata(file_name_strip_2_of_4,0,header=False)
img_processed_stripe_2 = shave_and_rotate(image_stripe_2,angle=13.43)

image_stripe_3 = fits.getdata(file_name_strip_3_of_4,0,header=False)
img_processed_stripe_3 = shave_and_rotate(image_stripe_3,angle=6.63)

image_stripe_4 = fits.getdata(file_name_strip_4_of_4,0,header=False)
img_processed_stripe_4 = shave_and_rotate(image_stripe_4,angle=-0.04)


# In[10]:


# find the cross-sections and marginalizations

marginalization_dict = {}
cross_sec_dict = {}

marginalization_dict["strip_0"] = np.sum(img_processed_stripe_0,axis=0)
marginalization_dict["strip_1"] = np.sum(img_processed_stripe_1,axis=0)
marginalization_dict["strip_2"] = np.sum(img_processed_stripe_2,axis=0)
marginalization_dict["strip_3"] = np.sum(img_processed_stripe_3,axis=0)
marginalization_dict["strip_4"] = np.sum(img_processed_stripe_4,axis=0)

cross_sec_dict["strip_0"] = img_processed_stripe_0[int(0.5*np.shape(img_processed_stripe_0)[0]),:]
cross_sec_dict["strip_1"] = img_processed_stripe_1[int(0.5*np.shape(img_processed_stripe_1)[0]),:]
cross_sec_dict["strip_2"] = img_processed_stripe_2[int(0.5*np.shape(img_processed_stripe_2)[0]),:]
cross_sec_dict["strip_3"] = img_processed_stripe_3[int(0.5*np.shape(img_processed_stripe_3)[0]),:]
cross_sec_dict["strip_4"] = img_processed_stripe_4[int(0.5*np.shape(img_processed_stripe_4)[0]),:]


# In[21]:


############################
# BEGIN USER INPUTS 2 OF 2
############################

image_injected_planet = img_processed_stripe_0
cross_sec_injected_planet = cross_sec_dict["strip_0"]
marginalization_injected_planet = marginalization_dict["strip_0"]

############################
# END USER INPUTS 1 OF 2
############################


# In[27]:


planet_loc_pix = np.divide(dist_asec,0.0107)

f, ((ax1, ax2, ax3, ax4, ax5, ax6), (ax7, ax8, ax9, ax10, ax11, ax12)) = plt.subplots(2, 6, figsize=(24, 16))

# top row: 2D color plot and cross-sections
# bottom row: marginalizations

# top left: 2D color plot
subplot1 = ax1.imshow(image_injected_planet, origin="lower", aspect="auto", vmin=-5000, vmax=5000)
ax1.axvline(x=0.5*np.shape(image_injected_planet)[0]-planet_loc_pix,
            linestyle=":", color="k", linewidth=4, alpha=0.4)
#plt.colorbar(subplot1)

# plot cross-sections and their differences between different strips
ax2.plot(cross_sec_dict["strip_0"], label="cross sec")
ax2.plot(np.subtract(cross_sec_injected_planet,cross_sec_dict["strip_0"]), label="diff")
ax2.axvline(x=0.5*np.shape(image_injected_planet)[0]-planet_loc_pix, linestyle=":", color="k", linewidth=4, alpha=0.4)
strip_0_ks_cross_sec = do_KS(cross_sec_injected_planet,cross_sec_dict["strip_0"])
ax2.legend()
ax2.set_title("Cross-sec rel. to strip 0\nD = "
              + str(np.round(strip_0_ks_cross_sec[0],4))
              + ",\nval_crit = " + str(np.round(strip_0_ks_cross_sec[1],4)) + ",\np_val = " + str(np.round(strip_0_ks_cross_sec[2],4)))

ax3.plot(cross_sec_dict["strip_1"], label="cross sec")
ax3.plot(np.subtract(cross_sec_injected_planet,cross_sec_dict["strip_1"]), label="diff")
ax3.axvline(x=0.5*np.shape(image_injected_planet)[0]-planet_loc_pix, linestyle=":", color="k", linewidth=4, alpha=0.4)
strip_1_ks_cross_sec = do_KS(cross_sec_injected_planet,cross_sec_dict["strip_1"])
ax3.legend()
ax3.set_title("Cross-sec rel. to strip 1\nD = "
              + str(np.round(strip_1_ks_cross_sec[0],4)) + ",\nval_crit = "
              + str(np.round(strip_1_ks_cross_sec[1],4)) + ",\np_val = " + str(np.round(strip_1_ks_cross_sec[2],4)))


ax4.plot(cross_sec_dict["strip_2"], label="cross sec")
ax4.plot(np.subtract(cross_sec_injected_planet,cross_sec_dict["strip_2"]), label="diff")
ax4.axvline(x=0.5*np.shape(image_injected_planet)[0]-planet_loc_pix, linestyle=":", color="k", linewidth=4, alpha=0.4)
strip_2_ks_cross_sec = do_KS(cross_sec_injected_planet,cross_sec_dict["strip_2"])
ax4.legend()
ax4.set_title("Cross-sec rel. to strip 2\nD = "
              + str(np.round(strip_2_ks_cross_sec[0],4)) + ",\nval_crit = "
              + str(np.round(strip_2_ks_cross_sec[1],4)) + ",\np_val = " + str(np.round(strip_2_ks_cross_sec[2],4)))


ax5.plot(cross_sec_dict["strip_3"], label="cross sec")
ax5.plot(np.subtract(cross_sec_injected_planet,cross_sec_dict["strip_3"]), label="diff")
ax5.axvline(x=0.5*np.shape(image_injected_planet)[0]-planet_loc_pix, linestyle=":", color="k", linewidth=4, alpha=0.4)
strip_3_ks_cross_sec = do_KS(cross_sec_injected_planet,cross_sec_dict["strip_3"])
ax5.legend()
ax5.set_title("Cross-sec rel. to strip 3\nD = "
              + str(np.round(strip_3_ks_cross_sec[0],4)) + ",\nval_crit = "
              + str(np.round(strip_3_ks_cross_sec[1],4)) + ",\np_val = " + str(np.round(strip_3_ks_cross_sec[2],4)))


ax6.plot(cross_sec_dict["strip_4"], label="cross sec")
ax6.plot(np.subtract(cross_sec_injected_planet,cross_sec_dict["strip_4"]), label="diff")
ax6.axvline(x=0.5*np.shape(image_injected_planet)[0]-planet_loc_pix, linestyle=":", color="k", linewidth=4, alpha=0.4)
strip_4_ks_cross_sec = do_KS(cross_sec_injected_planet,cross_sec_dict["strip_4"])
ax6.legend()
ax6.set_title("Cross-sec rel. to strip 4\nD = "
              + str(np.round(strip_4_ks_cross_sec[0],4)) + ",\nval_crit = "
              + str(np.round(strip_4_ks_cross_sec[1],4)) + ",\np_val = " + str(np.round(strip_4_ks_cross_sec[2],4)))


# bottom-left (ax7): blank

# plot cross-sections and their differences between different strips
ax8.plot(marginalization_dict["strip_0"], label="marginalization")
ax8.plot(np.subtract(marginalization_injected_planet,marginalization_dict["strip_0"]), label="diff")
ax8.axvline(x=0.5*np.shape(image_injected_planet)[0]-planet_loc_pix, linestyle=":", color="k", linewidth=4, alpha=0.4)
strip_0_ks_marg = do_KS(marginalization_injected_planet,marginalization_dict["strip_0"])
ax8.legend()
ax8.set_title("Marg rel. to strip 0\nD = "
              + str(np.round(strip_0_ks_marg[0],4)) + ",\nval_crit = "
              + str(np.round(strip_0_ks_marg[1],4)) + ",\np_val = " + str(np.round(strip_0_ks_marg[2],4)))

ax9.plot(marginalization_dict["strip_1"], label="marginalization")
ax9.plot(np.subtract(marginalization_injected_planet,marginalization_dict["strip_1"]), label="diff")
ax9.axvline(x=0.5*np.shape(image_injected_planet)[0]-planet_loc_pix, linestyle=":", color="k", linewidth=4, alpha=0.4)
strip_1_ks_marg = do_KS(marginalization_injected_planet,marginalization_dict["strip_1"])
ax9.legend()
ax9.set_title("Marg rel. to strip 1\nD = "
              + str(np.round(strip_1_ks_marg[0],4)) + ",\nval_crit = "
              + str(np.round(strip_1_ks_marg[1],4)) + ",\np_val = " + str(np.round(strip_1_ks_marg[2],4)))

ax10.plot(cross_sec_dict["strip_2"], label="marginalization")
ax10.plot(np.subtract(marginalization_injected_planet,marginalization_dict["strip_2"]), label="diff")
ax10.axvline(x=0.5*np.shape(image_injected_planet)[0]-planet_loc_pix, linestyle=":", color="k", linewidth=4, alpha=0.4)
strip_2_ks_marg = do_KS(marginalization_injected_planet,marginalization_dict["strip_2"])
ax10.legend()
ax10.set_title("Marg rel. to strip 2\nD = "
               + str(np.round(strip_2_ks_marg[0],4)) + ",\nval_crit = "
               + str(np.round(strip_2_ks_marg[1],4)) + ",\np_val = " + str(np.round(strip_2_ks_marg[2],4)))

ax11.plot(cross_sec_dict["strip_3"], label="marginalization")
ax11.plot(np.subtract(marginalization_injected_planet,marginalization_dict["strip_3"]), label="diff")
ax11.axvline(x=0.5*np.shape(image_injected_planet)[0]-planet_loc_pix, linestyle=":", color="k", linewidth=4, alpha=0.4)
strip_3_ks_marg = do_KS(marginalization_injected_planet,marginalization_dict["strip_3"])
ax11.legend()
ax11.set_title("Marg rel. to strip 3\nD = "
               + str(np.round(strip_3_ks_marg[0],4)) + ",\nval_crit = "
               + str(np.round(strip_3_ks_marg[1],4)) + ",\np_val = " + str(np.round(strip_3_ks_marg[2],4)))

ax12.plot(cross_sec_dict["strip_4"], label="marginalization")
ax12.plot(np.subtract(marginalization_injected_planet,marginalization_dict["strip_4"]), label="diff")
ax12.axvline(x=0.5*np.shape(image_injected_planet)[0]-planet_loc_pix, linestyle=":", color="k", linewidth=4, alpha=0.4)
strip_4_ks_marg = do_KS(marginalization_injected_planet,marginalization_dict["strip_4"])
ax12.legend()
ax12.set_title("Marg rel. to strip 4\nD = "
               + str(np.round(strip_4_ks_marg[0],4)) + ",\nval_crit = "
               + str(np.round(strip_4_ks_marg[1],4)) + ",\np_val = " + str(np.round(strip_4_ks_marg[2],4)))

#ax6.set_ylim([-400,700]) # for 0.01 companions
#ax3.set_ylim([-3000,6000]) # for 0.1 companions

#f.suptitle(plot_file_name_prefix + os.path.basename(file_name_array_choice[file_num]))
#plt.tight_layout()
plt.savefig(new_filename, dpi=150)
plt.close()
#plt.show()

print("Saved " + new_filename)