import urllib
import configparser
import numpy as np
import pandas as pd
import glob
import os
from scipy import ndimage, misc, stats
from astropy.io import fits
import matplotlib.pyplot as plt


def shave_and_rotate(img, angle):
    '''
    Shave off edges of frames (to get rid of the NaNs) and rotate
    such that E (or N) is on the left, W (or S) on the right
    '''

    # shave off this much of the edges
    edge_size_shave = 150

    # shave off edges symmetrically
    img_shaved = img[edge_size_shave:int(np.shape(img)[0]-edge_size_shave),
                     edge_size_shave:int(np.shape(img)[1]-edge_size_shave)]

    img_shaved_rotated = ndimage.rotate(img_shaved, angle, reshape=False)

    return img_shaved_rotated


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
        if ((np.round(planet_current_ampl,6) == np.round(comp_ampl_pass,6)) and
            (np.round(planet_loc_asec,3) == np.round(dist_asec_pass,3))):

            file_name_this_strip_of_interest = file_names[file_num]
            found_one_ticker += 1

    if (found_one_ticker > 1):

        print("Something is wrong-- found more than 1 matching name!")

    elif (found_one_ticker == 0):

        print("No matching files for comp_ampl " + str(comp_ampl_pass) + \
            ", dist_asec " + str(dist_asec_pass))
        #import ipdb; ipdb.set_trace()

    else:

        print("Found matching file \n" + str(file_name_this_strip_of_interest))

    return file_name_this_strip_of_interest


def return_residuals(array_loaded):
    '''
    Feed in an array and have two sets of residuals returned and ready for
    comparison

    INPUT:
    array_loaded: a derotated array arranged so that E (or N) is on the left, W (or S) on the right

    OUTPUTS:
    radial_median_array_E: median of residuals of the eastern (or northern) half-strip
    radial_median_array_W: " " " western (or southern) half-strip
    '''

    # remove ~zeros from consideration
    #array_relevant = np.copy(array_loaded)
    #array_relevant[np.abs(array_loaded)<1e-10] = np.nan
    # (the below is a kludge, since the above method was not working well)
    array_relevant = np.copy(array_loaded)
    array_relevant[:50,:] = np.nan
    array_relevant[57:,:] = np.nan
    #import ipdb; ipdb.set_trace()
    # convolve with a smoothing function? I've decided against it for now...

    # make grid to define center
    X,Y = np.meshgrid(np.arange(-0.5*np.shape(array_loaded)[1],0.5*np.shape(array_loaded)[1],1),
                  np.arange(-0.5*np.shape(array_loaded)[0],0.5*np.shape(array_loaded)[0],1))
    # remove centering error
    X = np.add(X,0.5)
    Y = np.add(Y,0.5)

    # initialize arrays to hold residuals
    arr_length = 50 # boundary of 50 is slightly short of the radius of the long-baseline masks
    radial_std_array_E = np.nan*np.ones(arr_length)
    radial_median_array_E = np.nan*np.ones(arr_length)
    radial_max_array_E = np.nan*np.ones(arr_length)
    radial_std_array_W = np.nan*np.ones(arr_length)
    radial_median_array_W = np.nan*np.ones(arr_length)
    radial_max_array_W = np.nan*np.ones(arr_length)

    # thickness of ring to use (in pixels)
    rad_thickness = 1

    # loop over radii, one pixel at a time
    for rad_central in range(0,arr_length):

        rad_inner = rad_central-0.5*rad_thickness
        rad_outer = rad_central+0.5*rad_thickness

        # select pixels
        mask_ring_bool = np.logical_and(np.power(X,2) + np.power(Y,2) <= np.power(rad_outer,2),
                            np.power(X,2) + np.power(Y,2) >= np.power(rad_inner,2))
        # cast boolean -> int
        mask_ring_int = 1*mask_ring_bool
        # cast zeros -> nans
        mask_ring_nans = np.copy(mask_ring_int).astype(float)
        mask_ring_nans[mask_ring_nans == 0] = np.nan

        # mask the array
        array_masked = np.multiply(array_relevant,mask_ring_nans)

        # chop into E (left) and W (right) halves
        # (note that the derotation should have been done beforehand)
        array_masked_E = array_masked[:,0:int(0.5*np.shape(array_loaded)[1])]
        array_masked_W = array_masked[:,int(0.5*np.shape(array_loaded)[1]):]
        #import ipdb; ipdb.set_trace()

        # extract data
        stdev_E = np.nanstd(array_masked_E)
        stdev_W = np.nanstd(array_masked_W)
        median_E = np.nanmedian(array_masked_E)
        median_W = np.nanmedian(array_masked_W)
        max_E = np.nanmax(array_masked_E)
        max_W = np.nanmax(array_masked_W)

        # populate arrays
        # (note that they are both in r-space, such that r:0->inf is left->right;
        # no additional flipping is necessary)
        radial_std_array_E[rad_central] = stdev_E
        radial_std_array_W[rad_central] = stdev_W
        radial_median_array_E[rad_central] = median_E
        radial_median_array_W[rad_central] = median_W
        radial_max_array_E[rad_central] = max_E
        radial_max_array_W[rad_central] = max_W

    return radial_median_array_E, radial_median_array_W


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

    # if one of the samples is just NaNs, just return NaNs
    if np.logical_and(not np.isfinite(np.nanmin(empirical_sample_1)),not np.isfinite(np.nanmax(empirical_sample_1))):
        return np.nan, np.nan, np.nan
    if np.logical_and(not np.isfinite(np.nanmin(empirical_sample_2)),not np.isfinite(np.nanmax(empirical_sample_2))):
        return np.nan, np.nan, np.nan

    return D, val_crit, p_val


def main(stripe_w_planet, half_w_planet, write_csv_basename):
    '''
    Read in arrays, process them, find residuals, calculate KS test, and
    write out the results as csv files (plus some FYI plots)

    INPUTS:
    stripe_w_planet: string which sets the strip with planets injected along the median angle
        (choices are ["0","1","2","3","4","0V","1V","2V","3V","4V"])
    half_w_planet: the East/West/North/South half of the stripe_w_planet which contains the
        planet (choices are [E,W])
    write_csv_basename: csv containing the data which will be written
    '''

    # configuration data
    config = configparser.ConfigParser() # for parsing values in .init file
    config.read("./modules/config.ini")

    # directory containing subdirectory of files of different stripes with injected planets
    stem_adi_frames_lambda_over_B = str(config["data_dirs"]["DIR_ADI_LAMBDA_B_W_PLANETS"])
    # directory of baseline frames with no injected planets
    stem_adi_frames_lambda_over_B_no_planets = str(config["data_dirs"]["DIR_ADI_LAMBDA_B_NO_PLANETS"])

    # make lots of lists of files for
    # - every set of planet injections (with two different azimuths around the host star)
    # - as imprinted in every one of the five stripes

    # first get baseline ADI frames with no injected planets at all
    file_name_strip_0_of_4_baseline_no_planet = stem_adi_frames_lambda_over_B_no_planets + \
        "strip_0_of_4_no_planets.fits"
    file_name_strip_1_of_4_baseline_no_planet = stem_adi_frames_lambda_over_B_no_planets + \
        "strip_1_of_4_no_planets.fits"
    file_name_strip_2_of_4_baseline_no_planet = stem_adi_frames_lambda_over_B_no_planets + \
        "strip_2_of_4_no_planets.fits"
    file_name_strip_3_of_4_baseline_no_planet = stem_adi_frames_lambda_over_B_no_planets + \
        "strip_3_of_4_no_planets.fits"
    file_name_strip_4_of_4_baseline_no_planet = stem_adi_frames_lambda_over_B_no_planets + \
        "strip_4_of_4_no_planets.fits"
    file_name_strip_0V_of_4V_baseline_no_planet = stem_adi_frames_lambda_over_B_no_planets + \
        "strip_0V_of_4V_no_planets.fits"
    file_name_strip_1V_of_4V_baseline_no_planet = stem_adi_frames_lambda_over_B_no_planets + \
        "strip_1V_of_4V_no_planets.fits"
    file_name_strip_2V_of_4V_baseline_no_planet = stem_adi_frames_lambda_over_B_no_planets + \
        "strip_2V_of_4V_no_planets.fits"
    file_name_strip_3V_of_4V_baseline_no_planet = stem_adi_frames_lambda_over_B_no_planets + \
        "strip_3V_of_4V_no_planets.fits"
    file_name_strip_4V_of_4V_baseline_no_planet = stem_adi_frames_lambda_over_B_no_planets + \
        "strip_4V_of_4V_no_planets.fits"
    ##import ipdb; ipdb.set_trace()
    # files where planets are injected along the strip 0 of 4, along 129.68 deg E of N)
    # glob of file names of ADI frames of A block strip 0 of 4
    file_names_strip_0_of_4_planetsInStrip0_50pt32_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_0_of_4_planetsInStrip0_50pt32_deg/*.fits"))
    # glob of file names of ADI frames of D block strip 1 of 4, with planets aligned with strip 0 along 129.68 deg E of N
    file_names_strip_1_of_4_planetsInStrip0_50pt32_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_1_of_4_planetsInStrip0_50pt32_deg/*.fits"))
    # glob of file names of ADI frames of D block strip 2 of 4, with planets aligned with strip 0 along 129.68 deg E of N
    file_names_strip_2_of_4_planetsInStrip0_50pt32_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_2_of_4_planetsInStrip0_50pt32_deg/*.fits"))
    # glob of file names of ADI frames of D block strip 3 of 4, with planets aligned with strip 0 along 129.68 deg E of N
    file_names_strip_3_of_4_planetsInStrip0_50pt32_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_3_of_4_planetsInStrip0_50pt32_deg/*.fits"))
    # glob of file names of ADI frames of D block strip 4 of 4, with planets aligned with strip 0 along 129.68 deg E of N
    file_names_strip_4_of_4_planetsInStrip0_50pt32_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_4_of_4_planetsInStrip0_50pt32_deg/*.fits"))
    # globs of files with planets aligned with strip 0 again, but along opposite
    # azimuth of 230.32 deg E of N
    file_names_strip_0_of_4_planetsInStrip0_230pt32_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_0_of_4_planetsInStrip0_230pt32_deg/*.fits"))
    file_names_strip_1_of_4_planetsInStrip0_230pt32_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_1_of_4_planetsInStrip0_230pt32_deg/*.fits"))
    file_names_strip_2_of_4_planetsInStrip0_230pt32_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_2_of_4_planetsInStrip0_230pt32_deg/*.fits"))
    file_names_strip_3_of_4_planetsInStrip0_230pt32_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_3_of_4_planetsInStrip0_230pt32_deg/*.fits"))
    file_names_strip_4_of_4_planetsInStrip0_230pt32_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_4_of_4_planetsInStrip0_230pt32_deg/*.fits"))
    # now North/South stripes
    file_names_strip_0V_of_4V_planetsInStrip0V_320pt32_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_0V_of_4V_planetsInStrip0V_320pt32_deg/*.fits"))
    file_names_strip_1V_of_4V_planetsInStrip0V_320pt32_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_1V_of_4V_planetsInStrip0V_320pt32_deg/*.fits"))
    file_names_strip_2V_of_4V_planetsInStrip0V_320pt32_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_2V_of_4V_planetsInStrip0V_320pt32_deg/*.fits"))
    file_names_strip_3V_of_4V_planetsInStrip0V_320pt32_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_3V_of_4V_planetsInStrip0V_320pt32_deg/*.fits"))
    file_names_strip_4V_of_4V_planetsInStrip0V_320pt32_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_4V_of_4V_planetsInStrip0V_320pt32_deg/*.fits"))
    file_names_strip_0V_of_4V_planetsInStrip0V_140pt32_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_0V_of_4V_planetsInStrip0V_140pt32_deg/*.fits"))
    file_names_strip_1V_of_4V_planetsInStrip0V_140pt32_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_1V_of_4V_planetsInStrip0V_140pt32_deg/*.fits"))
    file_names_strip_2V_of_4V_planetsInStrip0V_140pt32_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_2V_of_4V_planetsInStrip0V_140pt32_deg/*.fits"))
    file_names_strip_3V_of_4V_planetsInStrip0V_140pt32_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_3V_of_4V_planetsInStrip0V_140pt32_deg/*.fits"))
    file_names_strip_4V_of_4V_planetsInStrip0V_140pt32_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_4V_of_4V_planetsInStrip0V_140pt32_deg/*.fits"))
    ##import ipdb; ipdb.set_trace()
    # glob of file names of ADI frames with planets in strip 1 of 4, along 70.782 deg E of N and opposite 250.782 deg E of N
    file_names_strip_0_of_4_planetsInStrip1_70pt782_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_0_of_4_planetsInStrip1_70pt782_deg/*.fits"))
    file_names_strip_1_of_4_planetsInStrip1_70pt782_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_1_of_4_planetsInStrip1_70pt782_deg/*.fits"))
    file_names_strip_2_of_4_planetsInStrip1_70pt782_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_2_of_4_planetsInStrip1_70pt782_deg/*.fits"))
    file_names_strip_3_of_4_planetsInStrip1_70pt782_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_3_of_4_planetsInStrip1_70pt782_deg/*.fits"))
    file_names_strip_4_of_4_planetsInStrip1_70pt782_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_4_of_4_planetsInStrip1_70pt782_deg/*.fits"))
    file_names_strip_0_of_4_planetsInStrip1_250pt782_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_0_of_4_planetsInStrip1_250pt782_deg/*.fits"))
    file_names_strip_1_of_4_planetsInStrip1_250pt782_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_1_of_4_planetsInStrip1_250pt782_deg/*.fits"))
    file_names_strip_2_of_4_planetsInStrip1_250pt782_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_2_of_4_planetsInStrip1_250pt782_deg/*.fits"))
    file_names_strip_3_of_4_planetsInStrip1_250pt782_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_3_of_4_planetsInStrip1_250pt782_deg/*.fits"))
    file_names_strip_4_of_4_planetsInStrip1_250pt782_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_4_of_4_planetsInStrip1_250pt782_deg/*.fits"))
    # now North/South stripes
    file_names_strip_0V_of_4V_planetsInStrip1V_340pt782_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_0V_of_4V_planetsInStrip1V_340pt782_deg/*.fits"))
    file_names_strip_1V_of_4V_planetsInStrip1V_340pt782_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_1V_of_4V_planetsInStrip1V_340pt782_deg/*.fits"))
    file_names_strip_2V_of_4V_planetsInStrip1V_340pt782_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_2V_of_4V_planetsInStrip1V_340pt782_deg/*.fits"))
    file_names_strip_3V_of_4V_planetsInStrip1V_340pt782_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_3V_of_4V_planetsInStrip1V_340pt782_deg/*.fits"))
    file_names_strip_4V_of_4V_planetsInStrip1V_340pt782_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_4V_of_4V_planetsInStrip1V_340pt782_deg/*.fits"))
    file_names_strip_0V_of_4V_planetsInStrip1V_160pt782_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_0V_of_4V_planetsInStrip1V_160pt782_deg/*.fits"))
    file_names_strip_1V_of_4V_planetsInStrip1V_160pt782_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_1V_of_4V_planetsInStrip1V_160pt782_deg/*.fits"))
    file_names_strip_2V_of_4V_planetsInStrip1V_160pt782_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_2V_of_4V_planetsInStrip1V_160pt782_deg/*.fits"))
    file_names_strip_3V_of_4V_planetsInStrip1V_160pt782_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_3V_of_4V_planetsInStrip1V_160pt782_deg/*.fits"))
    file_names_strip_4V_of_4V_planetsInStrip1V_160pt782_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_4V_of_4V_planetsInStrip1V_160pt782_deg/*.fits"))
    ##import ipdb; ipdb.set_trace()
    # glob of file names of ADI frames with planets in strip 2 of 4, along 76.57 deg E of N and opposite 256.57 deg E of N
    file_names_strip_0_of_4_planetsInStrip2_76pt57_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_0_of_4_planetsInStrip2_76pt57_deg/*.fits"))
    file_names_strip_1_of_4_planetsInStrip2_76pt57_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_1_of_4_planetsInStrip2_76pt57_deg/*.fits"))
    file_names_strip_2_of_4_planetsInStrip2_76pt57_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_2_of_4_planetsInStrip2_76pt57_deg/*.fits"))
    file_names_strip_3_of_4_planetsInStrip2_76pt57_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_3_of_4_planetsInStrip2_76pt57_deg/*.fits"))
    file_names_strip_4_of_4_planetsInStrip2_76pt57_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_4_of_4_planetsInStrip2_76pt57_deg/*.fits"))
    file_names_strip_0_of_4_planetsInStrip2_256pt57_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_0_of_4_planetsInStrip2_256pt57_deg/*.fits"))
    file_names_strip_1_of_4_planetsInStrip2_256pt57_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_1_of_4_planetsInStrip2_256pt57_deg/*.fits"))
    file_names_strip_2_of_4_planetsInStrip2_256pt57_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_2_of_4_planetsInStrip2_256pt57_deg/*.fits"))
    file_names_strip_3_of_4_planetsInStrip2_256pt57_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_3_of_4_planetsInStrip2_256pt57_deg/*.fits"))
    file_names_strip_4_of_4_planetsInStrip2_256pt57_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_4_of_4_planetsInStrip2_256pt57_deg/*.fits"))
    # now North/South stripes
    file_names_strip_0V_of_4V_planetsInStrip2V_346pt57_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_0V_of_4V_planetsInStrip2V_346pt57_deg/*.fits"))
    file_names_strip_1V_of_4V_planetsInStrip2V_346pt57_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_1V_of_4V_planetsInStrip2V_346pt57_deg/*.fits"))
    file_names_strip_2V_of_4V_planetsInStrip2V_346pt57_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_2V_of_4V_planetsInStrip2V_346pt57_deg/*.fits"))
    file_names_strip_3V_of_4V_planetsInStrip2V_346pt57_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_3V_of_4V_planetsInStrip2V_346pt57_deg/*.fits"))
    file_names_strip_4V_of_4V_planetsInStrip2V_346pt57_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_4V_of_4V_planetsInStrip2V_346pt57_deg/*.fits"))
    file_names_strip_0V_of_4V_planetsInStrip2V_166pt57_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_0V_of_4V_planetsInStrip2V_166pt57_deg/*.fits"))
    file_names_strip_1V_of_4V_planetsInStrip2V_166pt57_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_1V_of_4V_planetsInStrip2V_166pt57_deg/*.fits"))
    file_names_strip_2V_of_4V_planetsInStrip2V_166pt57_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_2V_of_4V_planetsInStrip2V_166pt57_deg/*.fits"))
    file_names_strip_3V_of_4V_planetsInStrip2V_166pt57_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_3V_of_4V_planetsInStrip2V_166pt57_deg/*.fits"))
    file_names_strip_4V_of_4V_planetsInStrip2V_166pt57_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_4V_of_4V_planetsInStrip2V_166pt57_deg/*.fits"))

    # glob of file names of ADI frames with planets in strip 3 of 4, along 83.37 deg E of N and opposite 263.37 deg E of N
    file_names_strip_0_of_4_planetsInStrip3_83pt37_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_0_of_4_planetsInStrip3_83pt37_deg/*.fits"))
    file_names_strip_1_of_4_planetsInStrip3_83pt37_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_1_of_4_planetsInStrip3_83pt37_deg/*.fits"))
    file_names_strip_2_of_4_planetsInStrip3_83pt37_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_2_of_4_planetsInStrip3_83pt37_deg/*.fits"))
    file_names_strip_3_of_4_planetsInStrip3_83pt37_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_3_of_4_planetsInStrip3_83pt37_deg/*.fits"))
    file_names_strip_4_of_4_planetsInStrip3_83pt37_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_4_of_4_planetsInStrip3_83pt37_deg/*.fits"))
    file_names_strip_0_of_4_planetsInStrip3_263pt37_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_0_of_4_planetsInStrip3_263pt37_deg/*.fits"))
    file_names_strip_1_of_4_planetsInStrip3_263pt37_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_1_of_4_planetsInStrip3_263pt37_deg/*.fits"))
    file_names_strip_2_of_4_planetsInStrip3_263pt37_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_2_of_4_planetsInStrip3_263pt37_deg/*.fits"))
    file_names_strip_3_of_4_planetsInStrip3_263pt37_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_3_of_4_planetsInStrip3_263pt37_deg/*.fits"))
    file_names_strip_4_of_4_planetsInStrip3_263pt37_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_4_of_4_planetsInStrip3_263pt37_deg/*.fits"))
    # now North/South stripes
    file_names_strip_0V_of_4V_planetsInStrip3V_353pt37_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_0V_of_4V_planetsInStrip3V_353pt37_deg/*.fits"))
    file_names_strip_1V_of_4V_planetsInStrip3V_353pt37_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_1V_of_4V_planetsInStrip3V_353pt37_deg/*.fits"))
    file_names_strip_2V_of_4V_planetsInStrip3V_353pt37_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_2V_of_4V_planetsInStrip3V_353pt37_deg/*.fits"))
    file_names_strip_3V_of_4V_planetsInStrip3V_353pt37_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_3V_of_4V_planetsInStrip3V_353pt37_deg/*.fits"))
    file_names_strip_4V_of_4V_planetsInStrip3V_353pt37_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_4V_of_4V_planetsInStrip3V_353pt37_deg/*.fits"))
    file_names_strip_0V_of_4V_planetsInStrip3V_173pt37_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_0V_of_4V_planetsInStrip3V_173pt37_deg/*.fits"))
    file_names_strip_1V_of_4V_planetsInStrip3V_173pt37_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_1V_of_4V_planetsInStrip3V_173pt37_deg/*.fits"))
    file_names_strip_2V_of_4V_planetsInStrip3V_173pt37_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_2V_of_4V_planetsInStrip3V_173pt37_deg/*.fits"))
    file_names_strip_3V_of_4V_planetsInStrip3V_173pt37_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_3V_of_4V_planetsInStrip3V_173pt37_deg/*.fits"))
    file_names_strip_4V_of_4V_planetsInStrip3V_173pt37_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_4V_of_4V_planetsInStrip3V_173pt37_deg/*.fits"))

    # glob of file names of ADI frames with planets in strip 4 of 4, along 90.04 deg E of N and opposite 270.04 deg E of N
    file_names_strip_0_of_4_planetsInStrip4_90pt04_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_0_of_4_planetsInStrip4_90pt04_deg/*.fits"))
    file_names_strip_1_of_4_planetsInStrip4_90pt04_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_1_of_4_planetsInStrip4_90pt04_deg/*.fits"))
    file_names_strip_2_of_4_planetsInStrip4_90pt04_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_2_of_4_planetsInStrip4_90pt04_deg/*.fits"))
    file_names_strip_3_of_4_planetsInStrip4_90pt04_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_3_of_4_planetsInStrip4_90pt04_deg/*.fits"))
    file_names_strip_4_of_4_planetsInStrip4_90pt04_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_4_of_4_planetsInStrip4_90pt04_deg/*.fits"))
    file_names_strip_0_of_4_planetsInStrip4_270pt04_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_0_of_4_planetsInStrip4_270pt04_deg/*.fits"))
    file_names_strip_1_of_4_planetsInStrip4_270pt04_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_1_of_4_planetsInStrip4_270pt04_deg/*.fits"))
    file_names_strip_2_of_4_planetsInStrip4_270pt04_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_2_of_4_planetsInStrip4_270pt04_deg/*.fits"))
    file_names_strip_3_of_4_planetsInStrip4_270pt04_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_3_of_4_planetsInStrip4_270pt04_deg/*.fits"))
    file_names_strip_4_of_4_planetsInStrip4_270pt04_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_4_of_4_planetsInStrip4_270pt04_deg/*.fits"))
    # now North/South stripes
    file_names_strip_0V_of_4V_planetsInStrip4V_0pt04_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_0V_of_4V_planetsInStrip4V_0pt04_deg/*.fits"))
    file_names_strip_1V_of_4V_planetsInStrip4V_0pt04_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_1V_of_4V_planetsInStrip4V_0pt04_deg/*.fits"))
    file_names_strip_2V_of_4V_planetsInStrip4V_0pt04_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_2V_of_4V_planetsInStrip4V_0pt04_deg/*.fits"))
    file_names_strip_3V_of_4V_planetsInStrip4V_0pt04_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_3V_of_4V_planetsInStrip4V_0pt04_deg/*.fits"))
    file_names_strip_4V_of_4V_planetsInStrip4V_0pt04_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_4V_of_4V_planetsInStrip4V_0pt04_deg/*.fits"))
    file_names_strip_0V_of_4V_planetsInStrip4V_180pt04_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_0V_of_4V_planetsInStrip4V_180pt04_deg/*.fits"))
    file_names_strip_1V_of_4V_planetsInStrip4V_180pt04_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_1V_of_4V_planetsInStrip4V_180pt04_deg/*.fits"))
    file_names_strip_2V_of_4V_planetsInStrip4V_180pt04_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_2V_of_4V_planetsInStrip4V_180pt04_deg/*.fits"))
    file_names_strip_3V_of_4V_planetsInStrip4V_180pt04_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_3V_of_4V_planetsInStrip4V_180pt04_deg/*.fits"))
    file_names_strip_4V_of_4V_planetsInStrip4V_180pt04_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_4V_of_4V_planetsInStrip4V_180pt04_deg/*.fits"))
    ##import ipdb; ipdb.set_trace()
    # choose the arrays to use in the analysis
    if np.logical_and((stripe_w_planet == "0"),(half_w_planet == "E")):
        # frames with planet along the E: East of North; i.e., 0<PA<180
        file_names_strip_0_of_4_E = file_names_strip_0_of_4_planetsInStrip0_50pt32_deg
        file_names_strip_1_of_4_E = file_names_strip_1_of_4_planetsInStrip0_50pt32_deg
        file_names_strip_2_of_4_E = file_names_strip_2_of_4_planetsInStrip0_50pt32_deg
        file_names_strip_3_of_4_E = file_names_strip_3_of_4_planetsInStrip0_50pt32_deg
        file_names_strip_4_of_4_E = file_names_strip_4_of_4_planetsInStrip0_50pt32_deg
        plot_string = "stripe_w_planet_0E_"
    elif np.logical_and((stripe_w_planet == "0"),(half_w_planet == "W")):
        # frames with planet along the W: West of North; i.e., 180<PA<360
        file_names_strip_0_of_4_W = file_names_strip_0_of_4_planetsInStrip0_230pt32_deg
        file_names_strip_1_of_4_W = file_names_strip_1_of_4_planetsInStrip0_230pt32_deg
        file_names_strip_2_of_4_W = file_names_strip_2_of_4_planetsInStrip0_230pt32_deg
        file_names_strip_3_of_4_W = file_names_strip_3_of_4_planetsInStrip0_230pt32_deg
        file_names_strip_4_of_4_W = file_names_strip_4_of_4_planetsInStrip0_230pt32_deg
        # for differentiating plot file names
        plot_string = "stripe_w_planet_0W_"
    elif np.logical_and((stripe_w_planet == "1"),(half_w_planet == "E")):
        # E: East of North; i.e., 0<PA<180
        file_names_strip_0_of_4_E = file_names_strip_0_of_4_planetsInStrip1_70pt782_deg
        file_names_strip_1_of_4_E = file_names_strip_1_of_4_planetsInStrip1_70pt782_deg
        file_names_strip_2_of_4_E = file_names_strip_2_of_4_planetsInStrip1_70pt782_deg
        file_names_strip_3_of_4_E = file_names_strip_3_of_4_planetsInStrip1_70pt782_deg
        file_names_strip_4_of_4_E = file_names_strip_4_of_4_planetsInStrip1_70pt782_deg
        # for differentiating plot file names
        plot_string = "stripe_w_planet_1E_"
    elif np.logical_and((stripe_w_planet == "1"),(half_w_planet == "W")):
        # W: West of North; i.e., 180<PA<360
        file_names_strip_0_of_4_W = file_names_strip_0_of_4_planetsInStrip1_250pt782_deg
        file_names_strip_1_of_4_W = file_names_strip_1_of_4_planetsInStrip1_250pt782_deg
        file_names_strip_2_of_4_W = file_names_strip_2_of_4_planetsInStrip1_250pt782_deg
        file_names_strip_3_of_4_W = file_names_strip_3_of_4_planetsInStrip1_250pt782_deg
        file_names_strip_4_of_4_W = file_names_strip_4_of_4_planetsInStrip1_250pt782_deg
        # for differentiating plot file names
        plot_string = "stripe_w_planet_1W_"
    elif np.logical_and((stripe_w_planet == "2"),(half_w_planet == "E")):
        # E: East of North; i.e., 0<PA<180
        file_names_strip_0_of_4_E = file_names_strip_0_of_4_planetsInStrip2_76pt57_deg
        file_names_strip_1_of_4_E = file_names_strip_1_of_4_planetsInStrip2_76pt57_deg
        file_names_strip_2_of_4_E = file_names_strip_2_of_4_planetsInStrip2_76pt57_deg
        file_names_strip_3_of_4_E = file_names_strip_3_of_4_planetsInStrip2_76pt57_deg
        file_names_strip_4_of_4_E = file_names_strip_4_of_4_planetsInStrip2_76pt57_deg
        # for differentiating plot file names
        plot_string = "stripe_w_planet_2E_"
    elif np.logical_and((stripe_w_planet == "2"),(half_w_planet == "W")):
        # W: West of North; i.e., 180<PA<360
        file_names_strip_0_of_4_W = file_names_strip_0_of_4_planetsInStrip2_256pt57_deg
        file_names_strip_1_of_4_W = file_names_strip_1_of_4_planetsInStrip2_256pt57_deg
        file_names_strip_2_of_4_W = file_names_strip_2_of_4_planetsInStrip2_256pt57_deg
        file_names_strip_3_of_4_W = file_names_strip_3_of_4_planetsInStrip2_256pt57_deg
        file_names_strip_4_of_4_W = file_names_strip_4_of_4_planetsInStrip2_256pt57_deg
        # for differentiating plot file names
        plot_string = "stripe_w_planet_2W_"
    elif np.logical_and((stripe_w_planet == "3"),(half_w_planet == "E")):
        # E: East of North; i.e., 0<PA<180
        file_names_strip_0_of_4_E = file_names_strip_0_of_4_planetsInStrip3_83pt37_deg
        file_names_strip_1_of_4_E = file_names_strip_1_of_4_planetsInStrip3_83pt37_deg
        file_names_strip_2_of_4_E = file_names_strip_2_of_4_planetsInStrip3_83pt37_deg
        file_names_strip_3_of_4_E = file_names_strip_3_of_4_planetsInStrip3_83pt37_deg
        file_names_strip_4_of_4_E = file_names_strip_4_of_4_planetsInStrip3_83pt37_deg
        # for differentiating plot file names
        plot_string = "stripe_w_planet_3E_"
    elif np.logical_and((stripe_w_planet == "3"),(half_w_planet == "W")):
        # W: West of North; i.e., 180<PA<360
        file_names_strip_0_of_4_W = file_names_strip_0_of_4_planetsInStrip3_263pt37_deg
        file_names_strip_1_of_4_W = file_names_strip_1_of_4_planetsInStrip3_263pt37_deg
        file_names_strip_2_of_4_W = file_names_strip_2_of_4_planetsInStrip3_263pt37_deg
        file_names_strip_3_of_4_W = file_names_strip_3_of_4_planetsInStrip3_263pt37_deg
        file_names_strip_4_of_4_W = file_names_strip_4_of_4_planetsInStrip3_263pt37_deg
        # for differentiating plot file names
        plot_string = "stripe_w_planet_3W_"
    elif np.logical_and((stripe_w_planet == "4"),(half_w_planet == "E")):
        # E: East of North; i.e., 0<PA<180
        file_names_strip_0_of_4_E = file_names_strip_0_of_4_planetsInStrip4_90pt04_deg
        file_names_strip_1_of_4_E = file_names_strip_1_of_4_planetsInStrip4_90pt04_deg
        file_names_strip_2_of_4_E = file_names_strip_2_of_4_planetsInStrip4_90pt04_deg
        file_names_strip_3_of_4_E = file_names_strip_3_of_4_planetsInStrip4_90pt04_deg
        file_names_strip_4_of_4_E = file_names_strip_4_of_4_planetsInStrip4_90pt04_deg
        # for differentiating plot file names
        plot_string = "stripe_w_planet_4E_"
    elif np.logical_and((stripe_w_planet == "4"),(half_w_planet == "W")):
        # W: West of North; i.e., 180<PA<360
        file_names_strip_0_of_4_W = file_names_strip_0_of_4_planetsInStrip4_270pt04_deg
        file_names_strip_1_of_4_W = file_names_strip_1_of_4_planetsInStrip4_270pt04_deg
        file_names_strip_2_of_4_W = file_names_strip_2_of_4_planetsInStrip4_270pt04_deg
        file_names_strip_3_of_4_W = file_names_strip_3_of_4_planetsInStrip4_270pt04_deg
        file_names_strip_4_of_4_W = file_names_strip_4_of_4_planetsInStrip4_270pt04_deg
        # for differentiating plot file names
        plot_string = "stripe_w_planet_4W_"
    elif np.logical_and((stripe_w_planet == "0V"),(half_w_planet == "N")):
        # frames with planet along the E: East of North; i.e., 0<PA<180
        file_names_strip_0V_of_4V_N = file_names_strip_0V_of_4V_planetsInStrip0V_320pt32_deg
        file_names_strip_1V_of_4V_N = file_names_strip_1V_of_4V_planetsInStrip0V_320pt32_deg
        file_names_strip_2V_of_4V_N = file_names_strip_2V_of_4V_planetsInStrip0V_320pt32_deg
        file_names_strip_3V_of_4V_N = file_names_strip_3V_of_4V_planetsInStrip0V_320pt32_deg
        file_names_strip_4V_of_4V_N = file_names_strip_4V_of_4V_planetsInStrip0V_320pt32_deg
        plot_string = "stripe_w_planet_0VN_"
    elif np.logical_and((stripe_w_planet == "0V"),(half_w_planet == "S")):
        # frames with planet along the W: West of North; i.e., 180<PA<360
        file_names_strip_0V_of_4V_S = file_names_strip_0V_of_4V_planetsInStrip0V_140pt32_deg
        file_names_strip_1V_of_4V_S = file_names_strip_1V_of_4V_planetsInStrip0V_140pt32_deg
        file_names_strip_2V_of_4V_S = file_names_strip_2V_of_4V_planetsInStrip0V_140pt32_deg
        file_names_strip_3V_of_4V_S = file_names_strip_3V_of_4V_planetsInStrip0V_140pt32_deg
        file_names_strip_4V_of_4V_S = file_names_strip_4V_of_4V_planetsInStrip0V_140pt32_deg
        # for differentiating plot file names
        plot_string = "stripe_w_planet_0VS_"
    elif np.logical_and((stripe_w_planet == "1V"),(half_w_planet == "N")):
        # E: East of North; i.e., 0<PA<180
        file_names_strip_0V_of_4V_N = file_names_strip_0V_of_4V_planetsInStrip1V_340pt782_deg
        file_names_strip_1V_of_4V_N = file_names_strip_1V_of_4V_planetsInStrip1V_340pt782_deg
        file_names_strip_2V_of_4V_N = file_names_strip_2V_of_4V_planetsInStrip1V_340pt782_deg
        file_names_strip_3V_of_4V_N = file_names_strip_3V_of_4V_planetsInStrip1V_340pt782_deg
        file_names_strip_4V_of_4V_N = file_names_strip_4V_of_4V_planetsInStrip1V_340pt782_deg
        # for differentiating plot file names
        plot_string = "stripe_w_planet_1VN_"
    elif np.logical_and((stripe_w_planet == "1V"),(half_w_planet == "S")):
        # W: West of North; i.e., 180<PA<360
        file_names_strip_0V_of_4V_S = file_names_strip_0V_of_4V_planetsInStrip1V_160pt782_deg
        file_names_strip_1V_of_4V_S = file_names_strip_1V_of_4V_planetsInStrip1V_160pt782_deg
        file_names_strip_2V_of_4V_S = file_names_strip_2V_of_4V_planetsInStrip1V_160pt782_deg
        file_names_strip_3V_of_4V_S = file_names_strip_3V_of_4V_planetsInStrip1V_160pt782_deg
        file_names_strip_4V_of_4V_S = file_names_strip_4V_of_4V_planetsInStrip1V_160pt782_deg
        # for differentiating plot file names
        plot_string = "stripe_w_planet_1VS_"
    elif np.logical_and((stripe_w_planet == "2V"),(half_w_planet == "N")):
        # E: East of North; i.e., 0<PA<180
        file_names_strip_0V_of_4V_N = file_names_strip_0V_of_4V_planetsInStrip2V_346pt57_deg
        file_names_strip_1V_of_4V_N = file_names_strip_1V_of_4V_planetsInStrip2V_346pt57_deg
        file_names_strip_2V_of_4V_N = file_names_strip_2V_of_4V_planetsInStrip2V_346pt57_deg
        file_names_strip_3V_of_4V_N = file_names_strip_3V_of_4V_planetsInStrip2V_346pt57_deg
        file_names_strip_4V_of_4V_N = file_names_strip_4V_of_4V_planetsInStrip2V_346pt57_deg
        # for differentiating plot file names
        plot_string = "stripe_w_planet_2VN_"
    elif np.logical_and((stripe_w_planet == "2V"),(half_w_planet == "S")):
        # W: West of North; i.e., 180<PA<360
        file_names_strip_0V_of_4V_S = file_names_strip_0V_of_4V_planetsInStrip2V_166pt57_deg
        file_names_strip_1V_of_4V_S = file_names_strip_1V_of_4V_planetsInStrip2V_166pt57_deg
        file_names_strip_2V_of_4V_S = file_names_strip_2V_of_4V_planetsInStrip2V_166pt57_deg
        file_names_strip_3V_of_4V_S = file_names_strip_3V_of_4V_planetsInStrip2V_166pt57_deg
        file_names_strip_4V_of_4V_S = file_names_strip_4V_of_4V_planetsInStrip2V_166pt57_deg
        # for differentiating plot file names
        plot_string = "stripe_w_planet_2VS_"
    elif np.logical_and((stripe_w_planet == "3V"),(half_w_planet == "N")):
        # E: East of North; i.e., 0<PA<180
        file_names_strip_0V_of_4V_N = file_names_strip_0V_of_4V_planetsInStrip3V_353pt37_deg
        file_names_strip_1V_of_4V_N = file_names_strip_1V_of_4V_planetsInStrip3V_353pt37_deg
        file_names_strip_2V_of_4V_N = file_names_strip_2V_of_4V_planetsInStrip3V_353pt37_deg
        file_names_strip_3V_of_4V_N = file_names_strip_3V_of_4V_planetsInStrip3V_353pt37_deg
        file_names_strip_4V_of_4V_N = file_names_strip_4V_of_4V_planetsInStrip3V_353pt37_deg
        # for differentiating plot file names
        plot_string = "stripe_w_planet_3VN_"
    elif np.logical_and((stripe_w_planet == "3V"),(half_w_planet == "S")):
        # W: West of North; i.e., 180<PA<360
        file_names_strip_0V_of_4V_S = file_names_strip_0V_of_4V_planetsInStrip3V_173pt37_deg
        file_names_strip_1V_of_4V_S = file_names_strip_1V_of_4V_planetsInStrip3V_173pt37_deg
        file_names_strip_2V_of_4V_S = file_names_strip_2V_of_4V_planetsInStrip3V_173pt37_deg
        file_names_strip_3V_of_4V_S = file_names_strip_3V_of_4V_planetsInStrip3V_173pt37_deg
        file_names_strip_4V_of_4V_S = file_names_strip_4V_of_4V_planetsInStrip3V_173pt37_deg
        # for differentiating plot file names
        plot_string = "stripe_w_planet_3VS_"
    elif np.logical_and((stripe_w_planet == "4V"),(half_w_planet == "N")):
        # E: East of North; i.e., 0<PA<180
        file_names_strip_0V_of_4V_N = file_names_strip_0V_of_4V_planetsInStrip4V_0pt04_deg
        file_names_strip_1V_of_4V_N = file_names_strip_1V_of_4V_planetsInStrip4V_0pt04_deg
        file_names_strip_2V_of_4V_N = file_names_strip_2V_of_4V_planetsInStrip4V_0pt04_deg
        file_names_strip_3V_of_4V_N = file_names_strip_3V_of_4V_planetsInStrip4V_0pt04_deg
        file_names_strip_4V_of_4V_N = file_names_strip_4V_of_4V_planetsInStrip4V_0pt04_deg
        # for differentiating plot file names
        plot_string = "stripe_w_planet_4VN_"
    elif np.logical_and((stripe_w_planet == "4V"),(half_w_planet == "S")):
        # W: West of North; i.e., 180<PA<360
        file_names_strip_0V_of_4V_S = file_names_strip_0V_of_4V_planetsInStrip4V_180pt04_deg
        file_names_strip_1V_of_4V_S = file_names_strip_1V_of_4V_planetsInStrip4V_180pt04_deg
        file_names_strip_2V_of_4V_S = file_names_strip_2V_of_4V_planetsInStrip4V_180pt04_deg
        file_names_strip_3V_of_4V_S = file_names_strip_3V_of_4V_planetsInStrip4V_180pt04_deg
        file_names_strip_4V_of_4V_S = file_names_strip_4V_of_4V_planetsInStrip4V_180pt04_deg
        # for differentiating plot file names
        plot_string = "stripe_w_planet_4VS_"
    else:
        print("Don't know which lists of file names to use in the analysis!")
    ##import ipdb; ipdb.set_trace()
    # name of the plot for the publication outside the for-loop below
    lambda_over_B_pub_plot_filename_suffix = plot_string + "pub_plot.pdf"
    #import ipdb; ipdb.set_trace()
    # initialize DataFrame to hold KS test info
    col_names = ["dist_asec",
                "comp_ampl",
                "id_strip_w_planet",
                "id_opposite_to_half_w_planet",
                "D_strip_w_planets_rel_to_same_half_strip_wo_planet",
                "D_strip_w_planets_rel_to_other_half_same_strip_with_planet",
                "D_strip_w_planets_rel_to_strip_0",
                "D_strip_w_planets_rel_to_strip_1",
                "D_strip_w_planets_rel_to_strip_2",
                "D_strip_w_planets_rel_to_strip_3",
                "D_strip_w_planets_rel_to_strip_4",
                "val_crit_strip_w_planets_rel_to_strip_0",
                "val_crit_strip_w_planets_rel_to_strip_1",
                "val_crit_strip_w_planets_rel_to_strip_2",
                "val_crit_strip_w_planets_rel_to_strip_3",
                "val_crit_strip_w_planets_rel_to_strip_4"]
    ks_info_df = pd.DataFrame(columns = col_names)

    # generate lists of companion amplitudes and distances (asec) from host star
    # initialize DataFrame to hold data from ALL strips
    ## ## THIS MIGHT BE MORE WORK THAN IS WORTH IT FOR NOW
    '''
    df_params_all_strips = pd.DataFrame()
    strip_0_of_4_params =
    for t in range(0,len(file_names_strip_0_of_4_planetsInStrip0)):

        planet_current_ampl = float(file_names_strip_0_of_4_planetsInStrip0[t].split("_")[-2])
        planet_loc_asec = float(file_names_strip_0_of_4_planetsInStrip0[t].split("_")[-3])
        planet_deg_eofn = float(file_names_strip_0_of_4_planetsInStrip0[t].split("_")[-4])

        file_names_strip_0_of_4_planetsInStrip0
    '''

    # loop over each combination of injected companion amplitude and radial distance
    ## ## (THIS IS KIND OF CLUNKY; IMPROVE LATER IF TIME)
    #comp_ampl_array = np.array([0.0001,0.000251188643,0.000630957344,0.00158489319,0.005,0.01,0.02,0.03,0.05,0.1,0.251188643,1.])
    comp_ampl_array = np.array([0.005,0.01,0.02,0.03,0.05,0.1])
    dist_fwhm_array = np.array([0.1,0.4,0.7,1.,1.3,1.7,2.0,2.3,2.6,2.9,3.2,4.,5.])
    fwhm_pix = 9.728 # FWHM for 4.05um/8.25m, in pixels
    dist_pix_array = np.multiply(dist_fwhm_array,fwhm_pix)
    dist_asec_array = np.multiply(dist_pix_array,float(config["instrum_params"]["LMIR_PS"]))
    #import ipdb; ipdb.set_trace()
    # read in the baseline images with no planets
    image_baseline_stripe_0 = fits.getdata(file_name_strip_0_of_4_baseline_no_planet,0,header=False)
    baseline_processed_stripe_0 = shave_and_rotate(image_baseline_stripe_0,angle=-39.68)
    image_baseline_stripe_1 = fits.getdata(file_name_strip_1_of_4_baseline_no_planet,0,header=False)
    baseline_processed_stripe_1 = shave_and_rotate(image_baseline_stripe_1,angle=-19.218)
    image_baseline_stripe_2 = fits.getdata(file_name_strip_2_of_4_baseline_no_planet,0,header=False)
    baseline_processed_stripe_2 = shave_and_rotate(image_baseline_stripe_2,angle=-13.43)
    image_baseline_stripe_3 = fits.getdata(file_name_strip_3_of_4_baseline_no_planet,0,header=False)
    baseline_processed_stripe_3 = shave_and_rotate(image_baseline_stripe_3,angle=-6.63)
    image_baseline_stripe_4 = fits.getdata(file_name_strip_4_of_4_baseline_no_planet,0,header=False)
    baseline_processed_stripe_4 = shave_and_rotate(image_baseline_stripe_4,angle=0.04)

    image_baseline_stripe_0V = fits.getdata(file_name_strip_0V_of_4V_baseline_no_planet,0,header=False)
    baseline_processed_stripe_0V = shave_and_rotate(image_baseline_stripe_0V,angle=230.32)
    image_baseline_stripe_1V = fits.getdata(file_name_strip_1V_of_4V_baseline_no_planet,0,header=False)
    baseline_processed_stripe_1V = shave_and_rotate(image_baseline_stripe_1V,angle=250.782)
    image_baseline_stripe_2V = fits.getdata(file_name_strip_2V_of_4V_baseline_no_planet,0,header=False)
    baseline_processed_stripe_2V = shave_and_rotate(image_baseline_stripe_2V,angle=256.57)
    image_baseline_stripe_3V = fits.getdata(file_name_strip_3V_of_4V_baseline_no_planet,0,header=False)
    baseline_processed_stripe_3V = shave_and_rotate(image_baseline_stripe_3V,angle=263.37)
    image_baseline_stripe_4V = fits.getdata(file_name_strip_4V_of_4V_baseline_no_planet,0,header=False)
    baseline_processed_stripe_4V = shave_and_rotate(image_baseline_stripe_4V,angle=270.04)
    ##import ipdb; ipdb.set_trace()
    # loop over all pairs of (amplitude,radius); note that the half (E/W) of the
    # strips in which the planet with that (amplitude,radius) sits is set by
    # the half_w_planet
    for comp_ampl_num in range(0,len(comp_ampl_array)):
        for dist_asec_num in range(0,len(dist_asec_array)):

            comp_ampl = comp_ampl_array[comp_ampl_num]
            dist_asec = dist_asec_array[dist_asec_num]
            ##import ipdb; ipdb.set_trace()

            # name of the FYI plot to write
            new_filename = plot_string + \
                            "lambda_over_B_comp_ampl_" + str(comp_ampl) + \
                            "_dist_asec_" + str(dist_asec) + ".png"

            print("---------------------------------------------------")
            print("Doing KS test for comp_ampl " + str(comp_ampl) + \
                                " and dist_asec " + str(dist_asec))

            ## pluck out the interesting file names
            # frames with planets along eastern arm of strip
            ##import ipdb; ipdb.set_trace()
            if (half_w_planet == "E"):
                file_name_strip_0_of_4 = pluck_interesting_file_name(file_names_strip_0_of_4_E,
                                                                     comp_ampl_pass=comp_ampl,
                                                                     dist_asec_pass=dist_asec)
                image_stripe_0 = fits.getdata(file_name_strip_0_of_4,0,header=False)
                img_processed_stripe_0 = shave_and_rotate(image_stripe_0,angle=-39.68)
                file_name_strip_1_of_4 = pluck_interesting_file_name(file_names_strip_1_of_4_E,
                                                                     comp_ampl_pass=comp_ampl,
                                                                     dist_asec_pass=dist_asec)
                image_stripe_1 = fits.getdata(file_name_strip_1_of_4,0,header=False)
                img_processed_stripe_1 = shave_and_rotate(image_stripe_1,angle=-19.218)
                file_name_strip_2_of_4 = pluck_interesting_file_name(file_names_strip_2_of_4_E,
                                                                     comp_ampl_pass=comp_ampl,
                                                                     dist_asec_pass=dist_asec)
                image_stripe_2 = fits.getdata(file_name_strip_2_of_4,0,header=False)
                img_processed_stripe_2 = shave_and_rotate(image_stripe_2,angle=-13.43)
                file_name_strip_3_of_4 = pluck_interesting_file_name(file_names_strip_3_of_4_E,
                                                                     comp_ampl_pass=comp_ampl,
                                                                     dist_asec_pass=dist_asec)
                image_stripe_3 = fits.getdata(file_name_strip_3_of_4,0,header=False)
                img_processed_stripe_3 = shave_and_rotate(image_stripe_3,angle=-6.63)
                file_name_strip_4_of_4 = pluck_interesting_file_name(file_names_strip_4_of_4_E,
                                                                    comp_ampl_pass=comp_ampl,
                                                                    dist_asec_pass=dist_asec)
                image_stripe_4 = fits.getdata(file_name_strip_4_of_4,0,header=False)
                img_processed_stripe_4 = shave_and_rotate(image_stripe_4,angle=0.04)
            elif (half_w_planet == "W"):
                # frames with planets along western arm of strip
                file_name_strip_0_of_4 = pluck_interesting_file_name(file_names_strip_0_of_4_W,
                                                                     comp_ampl_pass=comp_ampl,
                                                                     dist_asec_pass=dist_asec)
                image_stripe_0 = fits.getdata(file_name_strip_0_of_4,0,header=False)
                img_processed_stripe_0 = shave_and_rotate(image_stripe_0,angle=-39.68)

                file_name_strip_1_of_4 = pluck_interesting_file_name(file_names_strip_1_of_4_W,
                                                                     comp_ampl_pass=comp_ampl,
                                                                     dist_asec_pass=dist_asec)
                image_stripe_1 = fits.getdata(file_name_strip_1_of_4,0,header=False)
                img_processed_stripe_1 = shave_and_rotate(image_stripe_1,angle=-19.218)

                file_name_strip_2_of_4 = pluck_interesting_file_name(file_names_strip_2_of_4_W,
                                                                     comp_ampl_pass=comp_ampl,
                                                                     dist_asec_pass=dist_asec)
                image_stripe_2 = fits.getdata(file_name_strip_2_of_4,0,header=False)
                img_processed_stripe_2 = shave_and_rotate(image_stripe_2,angle=-13.43)

                file_name_strip_3_of_4 = pluck_interesting_file_name(file_names_strip_3_of_4_W,
                                                                     comp_ampl_pass=comp_ampl,
                                                                     dist_asec_pass=dist_asec)
                image_stripe_3 = fits.getdata(file_name_strip_3_of_4,0,header=False)
                img_processed_stripe_3 = shave_and_rotate(image_stripe_3,angle=-6.63)

                file_name_strip_4_of_4 = pluck_interesting_file_name(file_names_strip_4_of_4_W,
                                                                    comp_ampl_pass=comp_ampl,
                                                                    dist_asec_pass=dist_asec)
                image_stripe_4 = fits.getdata(file_name_strip_4_of_4,0,header=False)
                img_processed_stripe_4 = shave_and_rotate(image_stripe_4,angle=0.04)
            elif (half_w_planet == "N"):
                # frames with planets along western arm of strip
                #import ipdb; ipdb.set_trace()
                file_name_strip_0V_of_4V = pluck_interesting_file_name(file_names_strip_0V_of_4V_N,
                                                                     comp_ampl_pass=comp_ampl,
                                                                     dist_asec_pass=dist_asec)
                image_stripe_0V = fits.getdata(file_name_strip_0V_of_4V,0,header=False)
                img_processed_stripe_0V = shave_and_rotate(image_stripe_0V,angle=230.32)

                file_name_strip_1V_of_4V = pluck_interesting_file_name(file_names_strip_1V_of_4V_N,
                                                                     comp_ampl_pass=comp_ampl,
                                                                     dist_asec_pass=dist_asec)
                image_stripe_1V = fits.getdata(file_name_strip_1V_of_4V,0,header=False)
                img_processed_stripe_1V = shave_and_rotate(image_stripe_1V,angle=250.782)

                file_name_strip_2V_of_4V = pluck_interesting_file_name(file_names_strip_2V_of_4V_N,
                                                                     comp_ampl_pass=comp_ampl,
                                                                     dist_asec_pass=dist_asec)
                image_stripe_2V = fits.getdata(file_name_strip_2V_of_4V,0,header=False)
                img_processed_stripe_2V = shave_and_rotate(image_stripe_2V,angle=256.57)

                file_name_strip_3V_of_4V = pluck_interesting_file_name(file_names_strip_3V_of_4V_N,
                                                                     comp_ampl_pass=comp_ampl,
                                                                     dist_asec_pass=dist_asec)
                image_stripe_3V = fits.getdata(file_name_strip_3V_of_4V,0,header=False)
                img_processed_stripe_3V = shave_and_rotate(image_stripe_3V,angle=263.37)

                file_name_strip_4V_of_4V = pluck_interesting_file_name(file_names_strip_4V_of_4V_N,
                                                                    comp_ampl_pass=comp_ampl,
                                                                    dist_asec_pass=dist_asec)
                image_stripe_4V = fits.getdata(file_name_strip_4V_of_4V,0,header=False)
                img_processed_stripe_4V = shave_and_rotate(image_stripe_4V,angle=270.04)
            elif (half_w_planet == "S"):
                # frames with planets along western arm of strip
                file_name_strip_0V_of_4V = pluck_interesting_file_name(file_names_strip_0V_of_4V_S,
                                                                     comp_ampl_pass=comp_ampl,
                                                                     dist_asec_pass=dist_asec)
                image_stripe_0V = fits.getdata(file_name_strip_0V_of_4V,0,header=False)
                img_processed_stripe_0V = shave_and_rotate(image_stripe_0V,angle=230.32)

                file_name_strip_1V_of_4V = pluck_interesting_file_name(file_names_strip_1V_of_4V_S,
                                                                     comp_ampl_pass=comp_ampl,
                                                                     dist_asec_pass=dist_asec)
                image_stripe_1V = fits.getdata(file_name_strip_1V_of_4V,0,header=False)
                img_processed_stripe_1V = shave_and_rotate(image_stripe_1V,angle=250.782)

                file_name_strip_2V_of_4V = pluck_interesting_file_name(file_names_strip_2V_of_4V_S,
                                                                     comp_ampl_pass=comp_ampl,
                                                                     dist_asec_pass=dist_asec)
                image_stripe_2V = fits.getdata(file_name_strip_2V_of_4V,0,header=False)
                img_processed_stripe_2V = shave_and_rotate(image_stripe_2V,angle=256.57)

                file_name_strip_3V_of_4V = pluck_interesting_file_name(file_names_strip_3V_of_4V_S,
                                                                     comp_ampl_pass=comp_ampl,
                                                                     dist_asec_pass=dist_asec)
                image_stripe_3V = fits.getdata(file_name_strip_3V_of_4V,0,header=False)
                img_processed_stripe_3V = shave_and_rotate(image_stripe_3V,angle=263.37)

                file_name_strip_4V_of_4V = pluck_interesting_file_name(file_names_strip_4V_of_4V_S,
                                                                    comp_ampl_pass=comp_ampl,
                                                                    dist_asec_pass=dist_asec)
                image_stripe_4V = fits.getdata(file_name_strip_4V_of_4V,0,header=False)
                img_processed_stripe_4V = shave_and_rotate(image_stripe_4V,angle=270.04)

            ##import ipdb; ipdb.set_trace()
            ## read in and process the images with planets
            # ... these are the frames with planets along eastern arm of strip

            #import ipdb; ipdb.set_trace()
            # find the cross-sections and marginalizations

            # initialize dictionaries
            marginalization_dict = {}
            residuals_dict = {}

            ## get residuals
            # baseline frames without planets: eastern halves and western halves
            # set down convention that host star is on the left side (i.e., 'east'
            # strips have to be flipped)

            # note that input to return_residuals is a derotated array arranged
            # so that E is on the left, W on the right
            baseline_processed_stripe_0_resids = return_residuals(baseline_processed_stripe_0)
            residuals_dict["baseline_strip_0_E"] = baseline_processed_stripe_0_resids[0]
            residuals_dict["baseline_strip_0_W"] = baseline_processed_stripe_0_resids[1]

            baseline_processed_stripe_1_resids = return_residuals(baseline_processed_stripe_1)
            residuals_dict["baseline_strip_1_E"] = baseline_processed_stripe_1_resids[0]
            residuals_dict["baseline_strip_1_W"] = baseline_processed_stripe_1_resids[1]

            baseline_processed_stripe_2_resids = return_residuals(baseline_processed_stripe_2)
            residuals_dict["baseline_strip_2_E"] = baseline_processed_stripe_2_resids[0]
            residuals_dict["baseline_strip_2_W"] = baseline_processed_stripe_2_resids[1]

            baseline_processed_stripe_3_resids = return_residuals(baseline_processed_stripe_3)
            residuals_dict["baseline_strip_3_E"] = baseline_processed_stripe_3_resids[0]
            residuals_dict["baseline_strip_3_W"] = baseline_processed_stripe_3_resids[1]

            baseline_processed_stripe_4_resids = return_residuals(baseline_processed_stripe_4)
            residuals_dict["baseline_strip_4_E"] = baseline_processed_stripe_4_resids[0]
            residuals_dict["baseline_strip_4_W"] = baseline_processed_stripe_4_resids[1]

            baseline_processed_stripe_0V_resids = return_residuals(baseline_processed_stripe_0V)
            residuals_dict["baseline_strip_0V_N"] = baseline_processed_stripe_0V_resids[0]
            residuals_dict["baseline_strip_0V_S"] = baseline_processed_stripe_0V_resids[1]

            baseline_processed_stripe_1V_resids = return_residuals(baseline_processed_stripe_1V)
            residuals_dict["baseline_strip_1V_N"] = baseline_processed_stripe_1V_resids[0]
            residuals_dict["baseline_strip_1V_S"] = baseline_processed_stripe_1V_resids[1]

            baseline_processed_stripe_2V_resids = return_residuals(baseline_processed_stripe_2V)
            residuals_dict["baseline_strip_2V_N"] = baseline_processed_stripe_2V_resids[0]
            residuals_dict["baseline_strip_2V_S"] = baseline_processed_stripe_2V_resids[1]

            baseline_processed_stripe_3V_resids = return_residuals(baseline_processed_stripe_3V)
            residuals_dict["baseline_strip_3V_N"] = baseline_processed_stripe_3V_resids[0]
            residuals_dict["baseline_strip_3V_S"] = baseline_processed_stripe_3V_resids[1]

            baseline_processed_stripe_4V_resids = return_residuals(baseline_processed_stripe_4V)
            residuals_dict["baseline_strip_4V_N"] = baseline_processed_stripe_4V_resids[0]
            residuals_dict["baseline_strip_4V_S"] = baseline_processed_stripe_4V_resids[1]
            ##import ipdb; ipdb.set_trace()
            # cross-sections of frames from half-stripes where a fake planet exists
            if np.logical_or((half_w_planet == "E"),(half_w_planet == "W")):
                img_processed_stripe_0_resids = return_residuals(img_processed_stripe_0)
                #fyi_image_stripe_0 = np.copy(img_processed_stripe_0) # for FYI plots
                residuals_dict["strip_0_E"] = img_processed_stripe_0_resids[0]
                residuals_dict["strip_0_W"] = img_processed_stripe_0_resids[1]
                img_processed_stripe_1_resids = return_residuals(img_processed_stripe_1)
                #fyi_image_stripe_1 = np.copy(img_processed_stripe_1) # for FYI plots
                residuals_dict["strip_1_E"] = img_processed_stripe_1_resids[0]
                residuals_dict["strip_1_W"] = img_processed_stripe_1_resids[1]
                img_processed_stripe_2_resids = return_residuals(img_processed_stripe_2)
                #fyi_image_stripe_2 = np.copy(img_processed_stripe_2) # for FYI plots
                residuals_dict["strip_2_E"] = img_processed_stripe_2_resids[0]
                residuals_dict["strip_2_W"] = img_processed_stripe_2_resids[1]
                img_processed_stripe_3_resids = return_residuals(img_processed_stripe_3)
                #fyi_image_stripe_3 = np.copy(img_processed_stripe_3) # for FYI plots
                residuals_dict["strip_3_E"] = img_processed_stripe_3_resids[0]
                residuals_dict["strip_3_W"] = img_processed_stripe_3_resids[1]
                img_processed_stripe_4_resids = return_residuals(img_processed_stripe_4)
                #fyi_image_stripe_4 = np.copy(img_processed_stripe_4) # for FYI plots
                residuals_dict["strip_4_E"] = img_processed_stripe_4_resids[0]
                residuals_dict["strip_4_W"] = img_processed_stripe_4_resids[1]
                residuals_dict["strip_0V_N"] = -9999
                residuals_dict["strip_0V_S"] = -9999
                residuals_dict["strip_1V_N"] = -9999
                residuals_dict["strip_1V_S"] = -9999
                residuals_dict["strip_2V_N"] = -9999
                residuals_dict["strip_2V_S"] = -9999
                residuals_dict["strip_3V_N"] = -9999
                residuals_dict["strip_3V_S"] = -9999
                residuals_dict["strip_4V_N"] = -9999
                residuals_dict["strip_4V_S"] = -9999
            elif np.logical_or((half_w_planet == "N"),(half_w_planet == "S")):
                residuals_dict["strip_0_E"] = -9999
                residuals_dict["strip_0_W"] = -9999
                residuals_dict["strip_1_E"] = -9999
                residuals_dict["strip_1_W"] = -9999
                residuals_dict["strip_2_E"] = -9999
                residuals_dict["strip_2_W"] = -9999
                residuals_dict["strip_3_E"] = -9999
                residuals_dict["strip_3_W"] = -9999
                residuals_dict["strip_4_E"] = -9999
                residuals_dict["strip_4_W"] = -9999
                img_processed_stripe_0V_resids = return_residuals(img_processed_stripe_0V)
                #fyi_image_stripe_0 = np.copy(img_processed_stripe_0V) # for FYI plots
                residuals_dict["strip_0V_N"] = img_processed_stripe_0V_resids[0]
                residuals_dict["strip_0V_S"] = img_processed_stripe_0V_resids[1]
                img_processed_stripe_1V_resids = return_residuals(img_processed_stripe_1V)
                #fyi_image_stripe_1 = np.copy(img_processed_stripe_1V) # for FYI plots
                residuals_dict["strip_1V_N"] = img_processed_stripe_1V_resids[0]
                residuals_dict["strip_1V_S"] = img_processed_stripe_1V_resids[1]
                img_processed_stripe_2V_resids = return_residuals(img_processed_stripe_2V)
                #fyi_image_stripe_2 = np.copy(img_processed_stripe_2V) # for FYI plots
                residuals_dict["strip_2V_N"] = img_processed_stripe_2V_resids[0]
                residuals_dict["strip_2V_S"] = img_processed_stripe_2V_resids[1]
                img_processed_stripe_3V_resids = return_residuals(img_processed_stripe_3V)
                #fyi_image_stripe_3 = np.copy(img_processed_stripe_3V) # for FYI plots
                residuals_dict["strip_3V_N"] = img_processed_stripe_3V_resids[0]
                residuals_dict["strip_3V_S"] = img_processed_stripe_3V_resids[1]
                img_processed_stripe_4V_resids = return_residuals(img_processed_stripe_4V)
                #fyi_image_stripe_4 = np.copy(img_processed_stripe_4V) # for FYI plots
                residuals_dict["strip_4V_N"] = img_processed_stripe_4V_resids[0]
                residuals_dict["strip_4V_S"] = img_processed_stripe_4V_resids[1]

            ##import ipdb; ipdb.set_trace()
            if np.logical_and((stripe_w_planet == "0"),(half_w_planet == "E")):
                # image of the injected stripe, with the planet to the right, for FYI plot
                # (note flipping, since the planet is 'east')
                image_injected_planet = np.flip(img_processed_stripe_0, axis=1)
                resids_1d_injected_planet = residuals_dict["strip_0_E"]
                resids_1d_baseline = residuals_dict["baseline_strip_0_E"]
                string_w_planet_indicator = "strip_0_E"
                string_opposite_indicator = "strip_0_W"
            elif np.logical_and((stripe_w_planet == "0"),(half_w_planet == "W")):
                image_injected_planet = img_processed_stripe_0
                resids_1d_injected_planet = residuals_dict["strip_0_W"]
                resids_1d_baseline = residuals_dict["baseline_strip_0_W"]
                string_w_planet_indicator = "strip_0_W"
                string_opposite_indicator = "strip_0_E"
            elif np.logical_and((stripe_w_planet == "1"),(half_w_planet == "E")):
                image_injected_planet = np.flip(img_processed_stripe_1, axis=1) # note flipping of 'east' half
                resids_1d_injected_planet = residuals_dict["strip_1_E"] # already flipped
                resids_1d_baseline = residuals_dict["baseline_strip_1_E"] # already flipped
                string_w_planet_indicator = "strip_1_E"
                string_opposite_indicator = "strip_1_W"
            elif np.logical_and((stripe_w_planet == "1"),(half_w_planet == "W")):
                image_injected_planet = img_processed_stripe_1
                resids_1d_injected_planet = residuals_dict["strip_1_W"]
                resids_1d_baseline = residuals_dict["baseline_strip_1_W"]
                string_w_planet_indicator = "strip_1_W"
                string_opposite_indicator = "strip_1_E"
            elif np.logical_and((stripe_w_planet == "2"),(half_w_planet == "E")):
                image_injected_planet = np.flip(img_processed_stripe_2, axis=1) # note flipping of 'east' half
                resids_1d_injected_planet = residuals_dict["strip_2_E"] # already flipped
                resids_1d_baseline = residuals_dict["baseline_strip_2_E"] # already flipped
                string_w_planet_indicator = "strip_2_E"
                string_opposite_indicator = "strip_2_W"
            elif np.logical_and((stripe_w_planet == "2"),(half_w_planet == "W")):
                image_injected_planet = img_processed_stripe_2
                resids_1d_injected_planet = residuals_dict["strip_2_W"]
                resids_1d_baseline = residuals_dict["baseline_strip_2_W"]
                string_w_planet_indicator = "strip_2_W"
                string_opposite_indicator = "strip_2_E"
            elif np.logical_and((stripe_w_planet == "3"),(half_w_planet == "E")):
                image_injected_planet = np.flip(img_processed_stripe_3, axis=1) # note flipping of 'east' half
                resids_1d_injected_planet = residuals_dict["strip_3_E"] # already flipped
                resids_1d_baseline = residuals_dict["baseline_strip_3_E"] # already flipped
                string_w_planet_indicator = "strip_3_E"
                string_opposite_indicator = "strip_3_W"
            elif np.logical_and((stripe_w_planet == "3"),(half_w_planet == "W")):
                image_injected_planet = img_processed_stripe_3
                resids_1d_injected_planet = residuals_dict["strip_3_W"]
                resids_1d_baseline = residuals_dict["baseline_strip_3_W"]
                string_w_planet_indicator = "strip_3_W"
                string_opposite_indicator = "strip_3_E"
            elif np.logical_and((stripe_w_planet == "4"),(half_w_planet == "E")):
                image_injected_planet = np.flip(img_processed_stripe_4, axis=1) # note flipping of 'east' half
                resids_1d_injected_planet = residuals_dict["strip_4_E"] # already flipped
                resids_1d_baseline = residuals_dict["baseline_strip_4_E"] # already flipped
                string_w_planet_indicator = "strip_4_E"
                string_opposite_indicator = "strip_4_W"
            elif np.logical_and((stripe_w_planet == "4"),(half_w_planet == "W")):
                image_injected_planet = img_processed_stripe_4
                resids_1d_injected_planet = residuals_dict["strip_4_W"]
                resids_1d_baseline = residuals_dict["baseline_strip_4_W"]
                string_w_planet_indicator = "strip_4_W"
                string_opposite_indicator = "strip_4_E"
            elif np.logical_and((stripe_w_planet == "0V"),(half_w_planet == "N")):
                # image of the injected stripe, with the planet to the right, for FYI plot
                # (note flipping, since the planet is 'east')
                image_injected_planet = np.flip(img_processed_stripe_0V, axis=1)
                resids_1d_injected_planet = residuals_dict["strip_0V_N"]
                resids_1d_baseline = residuals_dict["baseline_strip_0V_N"]
                string_w_planet_indicator = "strip_0V_N"
                string_opposite_indicator = "strip_0V_S"
            elif np.logical_and((stripe_w_planet == "0V"),(half_w_planet == "S")):
                image_injected_planet = img_processed_stripe_0V
                resids_1d_injected_planet = residuals_dict["strip_0V_S"]
                resids_1d_baseline = residuals_dict["baseline_strip_0V_S"]
                string_w_planet_indicator = "strip_0V_S"
                string_opposite_indicator = "strip_0V_N"
            elif np.logical_and((stripe_w_planet == "1V"),(half_w_planet == "N")):
                image_injected_planet = np.flip(img_processed_stripe_1V, axis=1) # note flipping of 'east' half
                resids_1d_injected_planet = residuals_dict["strip_1V_N"] # already flipped
                resids_1d_baseline = residuals_dict["baseline_strip_1V_N"] # already flipped
                string_w_planet_indicator = "strip_1V_N"
                string_opposite_indicator = "strip_1V_S"
            elif np.logical_and((stripe_w_planet == "1V"),(half_w_planet == "S")):
                image_injected_planet = img_processed_stripe_1V
                resids_1d_injected_planet = residuals_dict["strip_1V_S"]
                resids_1d_baseline = residuals_dict["baseline_strip_1V_S"]
                string_w_planet_indicator = "strip_1V_S"
                string_opposite_indicator = "strip_1V_N"
            elif np.logical_and((stripe_w_planet == "2V"),(half_w_planet == "N")):
                image_injected_planet = np.flip(img_processed_stripe_2V, axis=1) # note flipping of 'east' half
                resids_1d_injected_planet = residuals_dict["strip_2V_N"] # already flipped
                resids_1d_baseline = residuals_dict["baseline_strip_2V_N"] # already flipped
                string_w_planet_indicator = "strip_2V_N"
                string_opposite_indicator = "strip_2V_S"
            elif np.logical_and((stripe_w_planet == "2V"),(half_w_planet == "S")):
                image_injected_planet = img_processed_stripe_2V
                resids_1d_injected_planet = residuals_dict["strip_2V_S"]
                resids_1d_baseline = residuals_dict["baseline_strip_2V_S"]
                string_w_planet_indicator = "strip_2V_S"
                string_opposite_indicator = "strip_2V_N"
            elif np.logical_and((stripe_w_planet == "3V"),(half_w_planet == "N")):
                image_injected_planet = np.flip(img_processed_stripe_3V, axis=1) # note flipping of 'east' half
                resids_1d_injected_planet = residuals_dict["strip_3V_N"] # already flipped
                resids_1d_baseline = residuals_dict["baseline_strip_3V_N"] # already flipped
                string_w_planet_indicator = "strip_3V_N"
                string_opposite_indicator = "strip_3V_S"
            elif np.logical_and((stripe_w_planet == "3V"),(half_w_planet == "S")):
                image_injected_planet = img_processed_stripe_3V
                resids_1d_injected_planet = residuals_dict["strip_3V_S"]
                resids_1d_baseline = residuals_dict["baseline_strip_3V_S"]
                string_w_planet_indicator = "strip_3V_S"
                string_opposite_indicator = "strip_3V_N"
            elif np.logical_and((stripe_w_planet == "4V"),(half_w_planet == "N")):
                image_injected_planet = np.flip(img_processed_stripe_4V, axis=1) # note flipping of 'east' half
                resids_1d_injected_planet = residuals_dict["strip_4V_N"] # already flipped
                resids_1d_baseline = residuals_dict["baseline_strip_4V_N"] # already flipped
                string_w_planet_indicator = "strip_4V_N"
                string_opposite_indicator = "strip_4V_S"
            elif np.logical_and((stripe_w_planet == "4V"),(half_w_planet == "S")):
                image_injected_planet = img_processed_stripe_4V
                resids_1d_injected_planet = residuals_dict["strip_4V_S"]
                resids_1d_baseline = residuals_dict["baseline_strip_4V_S"]
                string_w_planet_indicator = "strip_4V_S"
                string_opposite_indicator = "strip_4V_N"
            else:
                print("No strip with planet specified!")
            ##import ipdb; ipdb.set_trace()
            ## calculate relevant quantities, put them into dataframe
            # KS statistic from cross-sections
            # baseline: comparison with same strip but without any planet at all
            ks_compare_w_baseline = do_KS(resids_1d_injected_planet,resids_1d_baseline)
            # opposite half: comparison with the *opposite* side of the strip
            # with the planet injected

            ks_compare_w_opposite = do_KS(resids_1d_injected_planet,residuals_dict[string_opposite_indicator])
            # others: different strips with planets along the same angle
            # (note one of these will be a repeat of strip_opposite_ks_cross_sec,
            # which will be removed further downstream in the pipeline)
            #residuals_dict[string_opposite_indicator] = np.multiply(np.nan,residuals_dict[string_opposite_indicator])
            if (half_w_planet == "E"):
                ks_compare_with_0 = do_KS(resids_1d_injected_planet,residuals_dict["strip_0_E"])
                ks_compare_with_1 = do_KS(resids_1d_injected_planet,residuals_dict["strip_1_E"])
                ks_compare_with_2 = do_KS(resids_1d_injected_planet,residuals_dict["strip_2_E"])
                ks_compare_with_3 = do_KS(resids_1d_injected_planet,residuals_dict["strip_3_E"])
                ks_compare_with_4 = do_KS(resids_1d_injected_planet,residuals_dict["strip_4_E"])
            elif (half_w_planet == "W"):
                ks_compare_with_0 = do_KS(resids_1d_injected_planet,residuals_dict["strip_0_W"])
                ks_compare_with_1 = do_KS(resids_1d_injected_planet,residuals_dict["strip_1_W"])
                ks_compare_with_2 = do_KS(resids_1d_injected_planet,residuals_dict["strip_2_W"])
                ks_compare_with_3 = do_KS(resids_1d_injected_planet,residuals_dict["strip_3_W"])
                ks_compare_with_4 = do_KS(resids_1d_injected_planet,residuals_dict["strip_4_W"])
            elif (half_w_planet == "N"):
                ks_compare_with_0 = do_KS(resids_1d_injected_planet,residuals_dict["strip_0V_N"])
                ks_compare_with_1 = do_KS(resids_1d_injected_planet,residuals_dict["strip_1V_N"])
                ks_compare_with_2 = do_KS(resids_1d_injected_planet,residuals_dict["strip_2V_N"])
                ks_compare_with_3 = do_KS(resids_1d_injected_planet,residuals_dict["strip_3V_N"])
                ks_compare_with_4 = do_KS(resids_1d_injected_planet,residuals_dict["strip_4V_N"])
            elif (half_w_planet == "S"):
                ks_compare_with_0 = do_KS(resids_1d_injected_planet,residuals_dict["strip_0V_S"])
                ks_compare_with_1 = do_KS(resids_1d_injected_planet,residuals_dict["strip_1V_S"])
                ks_compare_with_2 = do_KS(resids_1d_injected_planet,residuals_dict["strip_2V_S"])
                ks_compare_with_3 = do_KS(resids_1d_injected_planet,residuals_dict["strip_3V_S"])
                ks_compare_with_4 = do_KS(resids_1d_injected_planet,residuals_dict["strip_4V_S"])

            ##import ipdb; ipdb.set_trace()
            # put stats into a dictionary; note there are entries corresponding to
            # injected planets on the E and W arms of the strips
            my_dic = {"dist_asec": dist_asec,
                    "comp_ampl": comp_ampl,
                    "id_strip_w_planet": string_w_planet_indicator,
                    "id_opposite_to_half_w_planet": string_opposite_indicator,
                    "D_strip_w_planets_rel_to_same_half_strip_wo_planet": ks_compare_w_baseline[0],
                    "D_strip_w_planets_rel_to_other_half_same_strip_with_planet": ks_compare_w_opposite[0],
                    "D_strip_w_planets_rel_to_strip_0": ks_compare_with_0[0],
                    "D_strip_w_planets_rel_to_strip_1": ks_compare_with_1[0],
                    "D_strip_w_planets_rel_to_strip_2": ks_compare_with_2[0],
                    "D_strip_w_planets_rel_to_strip_3": ks_compare_with_3[0],
                    "D_strip_w_planets_rel_to_strip_4": ks_compare_with_4[0],
                    "val_crit_strip_w_planets_rel_to_strip_0": ks_compare_with_0[1],
                    "val_crit_strip_w_planets_rel_to_strip_1": ks_compare_with_1[1],
                    "val_crit_strip_w_planets_rel_to_strip_2": ks_compare_with_2[1],
                    "val_crit_strip_w_planets_rel_to_strip_3": ks_compare_with_3[1],
                    "val_crit_strip_w_planets_rel_to_strip_4": ks_compare_with_4[1]}

            ks_info_df.loc[len(ks_info_df)] = my_dic
            #import ipdb; ipdb.set_trace()
            '''
            print("dist_asec: " + str(np.round(dist_asec,3)))
            print("comp_ampl: " + str(np.round(comp_ampl,2)))
            print("strip_w_planets_rel_to_strip_0, cross-sec: " + str(strip_0_ks_cross_sec))
            print("strip_w_planets_rel_to_strip_0, marginalization: " + str(strip_0_ks_marg))
            print("strip_w_planets_rel_to_strip_1, cross-sec: " + str(strip_1_ks_cross_sec))
            print("strip_w_planets_rel_to_strip_1, marginalization: " + str(strip_1_ks_marg))
            print("strip_w_planets_rel_to_strip_2, cross-sec: " + str(strip_2_ks_cross_sec))
            print("strip_w_planets_rel_to_strip_2, marginalization: " + str(strip_2_ks_marg))
            print("strip_w_planets_rel_to_strip_3, cross-sec: " + str(strip_3_ks_cross_sec))
            print("strip_w_planets_rel_to_strip_3, marginalization: " + str(strip_3_ks_marg))
            print("strip_w_planets_rel_to_strip_4, cross-sec: " + str(strip_4_ks_cross_sec))
            print("strip_w_planets_rel_to_strip_4, marginalization: " + str(strip_4_ks_marg))
            '''

            ## some code to make discrete CDFs
            def discrete_cdf(input_array):
                N = len(input_array)
                X2 = np.sort(input_array)
                F2 = np.array(range(N))/float(N)
                return X2, F2

            ##############################################
            ## BEGIN GIANT BLOCK OF CODE TO MAKE A PLOT
            #import ipdb; ipdb.set_trace()
            # planet location in pixels to indicate in plot
            planet_loc_pix = np.divide(dist_asec,float(config["instrum_params"]["LMIR_PS"]))

            # make plot of the stripes
            f_2d, ((ax0_2d, ax1_2d, ax2_2d), (ax3_2d, ax4_2d, ax5_2d)) = plt.subplots(2, 3, figsize=(24, 36))
            if np.logical_or((half_w_planet == "E"),(half_w_planet == "W")):
                ax0_2d.imshow(img_processed_stripe_0, origin="lower", aspect="auto", vmin=-5000, vmax=5000)
                ax0_2d.set_title("img_processed_stripe_0")
                ax1_2d.imshow(img_processed_stripe_1, origin="lower", aspect="auto", vmin=-5000, vmax=5000)
                ax1_2d.set_title("img_processed_stripe_1")
                ax2_2d.imshow(img_processed_stripe_2, origin="lower", aspect="auto", vmin=-5000, vmax=5000)
                ax2_2d.set_title("img_processed_stripe_2")
                ax3_2d.imshow(img_processed_stripe_3, origin="lower", aspect="auto", vmin=-5000, vmax=5000)
                ax3_2d.set_title("img_processed_stripe_3")
                ax4_2d.imshow(img_processed_stripe_4, origin="lower", aspect="auto", vmin=-5000, vmax=5000)
                ax4_2d.set_title("img_processed_stripe_4")
            elif np.logical_or((half_w_planet == "N"),(half_w_planet == "S")):
                ax0_2d.imshow(img_processed_stripe_0V, origin="lower", aspect="auto", vmin=-5000, vmax=5000)
                ax0_2d.set_title("img_processed_stripe_0V")
                ax1_2d.imshow(img_processed_stripe_1V, origin="lower", aspect="auto", vmin=-5000, vmax=5000)
                ax1_2d.set_title("img_processed_stripe_1V")
                ax2_2d.imshow(img_processed_stripe_2V, origin="lower", aspect="auto", vmin=-5000, vmax=5000)
                ax2_2d.set_title("img_processed_stripe_2V")
                ax3_2d.imshow(img_processed_stripe_3V, origin="lower", aspect="auto", vmin=-5000, vmax=5000)
                ax3_2d.set_title("img_processed_stripe_3V")
                ax4_2d.imshow(img_processed_stripe_4V, origin="lower", aspect="auto", vmin=-5000, vmax=5000)
                ax4_2d.set_title("img_processed_stripe_4V")

            plt.suptitle("planet should be on left (E or N) or right (W or S)",fontsize=20)
            stripes_filename = "stripes_" + new_filename
            plt.savefig(stripes_filename, dpi=150)
            plt.close()
            print("Wrote FYI plot of stripes as " + stripes_filename)


            plt.clf()

            f, ((ax1, ax2, ax3, ax4, ax5, ax6, ax7),
                (ax1cdf, ax2cdf, ax3cdf, ax4cdf, ax5cdf, ax6cdf, ax7cdf)) = plt.subplots(2, 7, figsize=(24, 32))

            if (half_w_planet == "E"):
                residuals_moniker_0 = "strip_0_E"
                residuals_moniker_1 = "strip_1_E"
                residuals_moniker_2 = "strip_2_E"
                residuals_moniker_3 = "strip_3_E"
                residuals_moniker_4 = "strip_4_E"
            elif (half_w_planet == "W"):
                residuals_moniker_0 = "strip_0_W"
                residuals_moniker_1 = "strip_1_W"
                residuals_moniker_2 = "strip_2_W"
                residuals_moniker_3 = "strip_3_W"
                residuals_moniker_4 = "strip_4_W"
            elif (half_w_planet == "N"):
                residuals_moniker_0 = "strip_0V_N"
                residuals_moniker_1 = "strip_1V_N"
                residuals_moniker_2 = "strip_2V_N"
                residuals_moniker_3 = "strip_3V_N"
                residuals_moniker_4 = "strip_4V_N"
            elif (half_w_planet == "S"):
                residuals_moniker_0 = "strip_0V_S"
                residuals_moniker_1 = "strip_1V_S"
                residuals_moniker_2 = "strip_2V_S"
                residuals_moniker_3 = "strip_3V_S"
                residuals_moniker_4 = "strip_4V_S"

            # top row: 2D color plot and cross-sections
            # second row: CDFs
            #import ipdb; ipdb.set_trace()
            # top left: 2D color plot (which has been flipped if the planet is in the E half)
            subplot1 = ax1.imshow(image_injected_planet, origin="lower", aspect="auto", vmin=-5000, vmax=5000)
            ax1.axvline(x=0.5*np.shape(image_injected_planet)[0]+planet_loc_pix,
                        linestyle=":", color="k", linewidth=4, alpha=0.4)
            ax1.set_title("planet should be on RIGHT")
            #plt.colorbar(subplot1)

            # plot cross-sections and their differences between different strips
            ax2.plot(residuals_dict[residuals_moniker_0], label=residuals_moniker_0)
            ax2.plot(resids_1d_injected_planet, label="resids_1d_injected_planet")
            ax2.plot(np.subtract(resids_1d_injected_planet,residuals_dict[residuals_moniker_0]), label="diff")
            ax2.axvline(x=planet_loc_pix,
                linestyle=":", color="k", linewidth=4, alpha=0.4)
            ax2.set_ylim([-2000,2000])
            ax2.legend()
            ax2.set_title("Resids rel. to "+residuals_moniker_0+", E\nD = "
                          + str(np.round(ks_compare_with_0[0],4))
                          + ",\nval_crit = " + str(np.round(ks_compare_with_0[1],4))
                          + ",\np_val = " + str(np.round(ks_compare_with_0[2],4)))
            #import ipdb; ipdb.set_trace()

            # cdfs
            cdf_strip = discrete_cdf(residuals_dict[residuals_moniker_0])
            cdf_inj = discrete_cdf(resids_1d_injected_planet)
            ax2cdf.plot(cdf_strip[0], cdf_strip[1], label="strip cdf")
            ax2cdf.plot(cdf_inj[0], cdf_inj[1], label="injected cdf")
            #ax2cdf.plot(np.subtract(cdf_inj[1],cdf_strip[1]), label="diff")
            ax2cdf.legend()
            ax2cdf.set_title("CDF")

            ax3.plot(residuals_dict[residuals_moniker_1], label=residuals_moniker_1)
            ax3.plot(resids_1d_injected_planet, label="resids_1d_injected_planet")
            ax3.plot(np.subtract(resids_1d_injected_planet,residuals_dict[residuals_moniker_1]), label="diff")
            ax3.axvline(x=planet_loc_pix,
                linestyle=":", color="k", linewidth=4, alpha=0.4)
            ax3.set_ylim([-2000,2000])
            ax3.legend()
            ax3.set_title("Resids rel. to "+residuals_moniker_1+", E\nD = "
                          + str(np.round(ks_compare_with_1[0],4)) + ",\nval_crit = "
                          + str(np.round(ks_compare_with_1[1],4)) + ",\np_val = "
                          + str(np.round(ks_compare_with_1[2],4)))

            cdf_strip = discrete_cdf(residuals_dict[residuals_moniker_1])
            cdf_inj = discrete_cdf(resids_1d_injected_planet)
            ax3cdf.plot(cdf_strip[0], cdf_strip[1], label="strip cdf")
            ax3cdf.plot(cdf_inj[0], cdf_inj[1], label="injected cdf")
            #ax3cdf.plot(np.subtract(cdf_inj[1],cdf_strip[1]), label="diff")
            ax3cdf.legend()
            ax3cdf.set_title("CDF")


            ax4.plot(residuals_dict[residuals_moniker_2], label=residuals_moniker_2)
            ax4.plot(resids_1d_injected_planet, label="resids_1d_injected_planet")
            ax4.plot(np.subtract(resids_1d_injected_planet,residuals_dict[residuals_moniker_2]), label="diff")
            ax4.axvline(x=planet_loc_pix,
                linestyle=":", color="k", linewidth=4, alpha=0.4)
            ax4.set_ylim([-2000,2000])
            ax4.legend()
            ax4.set_title("Resids rel. to "+residuals_moniker_2+", E\nD = "
                          + str(np.round(ks_compare_with_2[0],4)) + ",\nval_crit = "
                          + str(np.round(ks_compare_with_2[1],4)) + ",\np_val = "
                          + str(np.round(ks_compare_with_2[2],4)))

            cdf_strip = discrete_cdf(residuals_dict[residuals_moniker_2])
            cdf_inj = discrete_cdf(resids_1d_injected_planet)
            ax4cdf.plot(cdf_strip[0], cdf_strip[1], label="strip cdf")
            ax4cdf.plot(cdf_inj[0], cdf_inj[1], label="injected cdf")
            #ax4cdf.plot(np.subtract(cdf_inj[1],cdf_strip[1]), label="diff")
            ax4cdf.legend()
            ax4cdf.set_title("CDF")


            ax5.plot(residuals_dict[residuals_moniker_3], label=residuals_moniker_3)
            ax5.plot(resids_1d_injected_planet, label="resids_1d_injected_planet")
            ax5.plot(np.subtract(resids_1d_injected_planet,residuals_dict[residuals_moniker_3]), label="diff")
            ax5.axvline(x=planet_loc_pix,
                linestyle=":", color="k", linewidth=4, alpha=0.4)
            ax5.set_ylim([-2000,2000])
            ax5.legend()
            ax5.set_title("Resids rel. to "+residuals_moniker_3+", E\nD = "
                          + str(np.round(ks_compare_with_3[0],4)) + ",\nval_crit = "
                          + str(np.round(ks_compare_with_3[1],4)) + ",\np_val = "
                          + str(np.round(ks_compare_with_3[2],4)))

            cdf_strip = discrete_cdf(residuals_dict[residuals_moniker_3])
            cdf_inj = discrete_cdf(resids_1d_injected_planet)
            ax5cdf.plot(cdf_strip[0], cdf_strip[1], label="strip cdf")
            ax5cdf.plot(cdf_inj[0], cdf_inj[1], label="injected cdf")
            #ax5cdf.plot(np.subtract(cdf_inj[1],cdf_strip[1]), label="diff")
            ax5cdf.legend()
            ax5cdf.set_title("CDF")


            ax6.plot(residuals_dict[residuals_moniker_4], label=residuals_moniker_4)
            ax6.plot(resids_1d_injected_planet, label="resids_1d_injected_planet")
            ax6.plot(np.subtract(resids_1d_injected_planet,residuals_dict[residuals_moniker_4]), label="diff")
            ax6.axvline(x=planet_loc_pix,
                linestyle=":", color="k", linewidth=4, alpha=0.4)
            ax6.set_ylim([-2000,2000])
            ax6.legend()
            ax6.set_title("Resids rel. to "+residuals_moniker_4+", E\nD = "
                          + str(np.round(ks_compare_with_4[0],4)) + ",\nval_crit = "
                          + str(np.round(ks_compare_with_4[1],4)) + ",\np_val = "
                          + str(np.round(ks_compare_with_4[2],4)))

            cdf_strip = discrete_cdf(residuals_dict[residuals_moniker_4])
            cdf_inj = discrete_cdf(resids_1d_injected_planet)
            ax6cdf.plot(cdf_strip[0], cdf_strip[1], label="strip cdf")
            ax6cdf.plot(cdf_inj[0], cdf_inj[1], label="injected cdf")
            #ax6cdf.plot(np.subtract(cdf_inj[1],cdf_strip[1]), label="diff")
            ax6cdf.legend()
            ax6cdf.set_title("CDF")


            # bottom-left (ax7): baseline
            ax7.plot(resids_1d_baseline, label="baseline w/o planet")
            ax7.plot(resids_1d_injected_planet, label="resids_1d_injected_planet")
            ax7.plot(np.subtract(resids_1d_injected_planet,resids_1d_baseline), label="diff")
            ax7.axvline(x=planet_loc_pix,
                linestyle=":", color="k", linewidth=4, alpha=0.4)
            ax7.set_ylim([-2000,2000])
            ax7.legend()
            ax7.set_title("Resids rel. to baseline, E\nD = "
                          + str(np.round(ks_compare_w_baseline[0],4)) + ",\nval_crit = "
                          + str(np.round(ks_compare_w_baseline[1],4)) + ",\np_val = "
                          + str(np.round(ks_compare_w_baseline[2],4)))

            #f.suptitle(plot_file_name_prefix + os.path.basename(file_name_array_choice[file_num]))
            #plt.tight_layout()
            plt.savefig(new_filename, dpi=150)
            plt.close()
            #plt.show()

            print("Saved " + new_filename)

            # make zoomed-in plot of one of the subplots
            cdf_strip = discrete_cdf(residuals_dict[residuals_moniker_3])
            cdf_inj = discrete_cdf(resids_1d_injected_planet)
            cdf_strip_interp = np.interp(cdf_inj[0],cdf_strip[0],cdf_strip[1]) # interpolate to find differences
            plt.plot(cdf_inj[0],cdf_strip_interp,label="strip in question")
            plt.plot(cdf_inj[0],cdf_inj[1],label="strip w planet")
            diff_abs = np.abs(np.subtract(cdf_strip_interp,cdf_inj[1]))
            max_index = np.nanargmax(diff_abs)
            plt.axvline(x=cdf_inj[0][max_index],linestyle=":")
            plt.plot(cdf_inj[0], diff_abs, label="diff")
            plt.xlim([-500,2500])
            plt.legend()
            plt.title(residuals_moniker_3 + "\nmax abs diff: " + str(diff_abs[max_index]))
            plt.savefig("zoom_in_" + new_filename)
            plt.close()
            print("Saved " + "zoom_in_" + new_filename)

    # taking all the data together, write it out as a csv
    ks_info_df.to_csv(write_csv_basename)
    print("Saved all data in " + write_csv_basename)

    # ... and make a plot for the publication
    '''
    # loop over each companion amplitude
    for comp_ampl_num in range(0,len(comp_ampl_array)):

        # select one companion amplitude
        ks_info_df_this_ampl = ks_info_df.where(
                                                np.round(ks_info_df["comp_ampl"],3) == np.round(comp_ampl_array[comp_ampl_num],3)
                                                )
        plt.clf()
        plt.plot(ks_info_df_this_ampl["dist_asec"], ks_info_df_this_ampl["D_xsec_strip_w_planets_rel_to_strip_0"],
            marker="o", label="Rel to strip 0")
        plt.plot(ks_info_df_this_ampl["dist_asec"], ks_info_df_this_ampl["D_xsec_strip_w_planets_rel_to_strip_1"],
            marker="o", label="Rel to strip 1")
        plt.plot(ks_info_df_this_ampl["dist_asec"], ks_info_df_this_ampl["D_xsec_strip_w_planets_rel_to_strip_2"],
            marker="o", label="Rel to strip 2")
        plt.plot(ks_info_df_this_ampl["dist_asec"], ks_info_df_this_ampl["D_xsec_strip_w_planets_rel_to_strip_3"],
            marker="o", label="Rel to strip 3")
        plt.plot(ks_info_df_this_ampl["dist_asec"], ks_info_df_this_ampl["D_xsec_strip_w_planets_rel_to_strip_4"],
            marker="o", label="Rel to strip 4")
        plt.xlabel("asec")
        plt.ylabel("KS D statistic")
        plt.axhline(y=np.nanmedian(ks_info_df_this_ampl["val_xsec_crit_strip_w_planets_rel_to_strip_0"]), linestyle=":", color="k")
        plt.legend()
        plt.title("KS test on cross-sections\ninjected companion amplitude: " +
                    str(np.round(comp_ampl_array[comp_ampl_num],3)))
        file_name_this = "ampl_" + str(np.round(comp_ampl_array[comp_ampl_num],3)) + \
            "_" + lambda_over_B_pub_plot_filename_suffix
        plt.savefig(file_name_this)
        plt.close()
        print("Saved lambda-over-B plot as " + file_name_this)
    '''
