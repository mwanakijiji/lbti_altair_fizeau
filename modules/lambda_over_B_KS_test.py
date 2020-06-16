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
        if ((np.round(planet_current_ampl,3) == np.round(comp_ampl_pass,3)) and
            (np.round(planet_loc_asec,3) == np.round(dist_asec_pass,3))):

            file_name_this_strip_of_interest = file_names[file_num]
            found_one_ticker += 1

    if (found_one_ticker > 1):

        print("Something is wrong-- found more than 1 matching name!")

    elif (found_one_ticker == 0):

        print("No matching files for comp_ampl " + str(comp_ampl_pass) + \
            ", dist_asec " + str(dist_asec_pass))
        import ipdb; ipdb.set_trace()

    else:

        print("Found matching file \n" + str(file_name_this_strip_of_interest))

    return file_name_this_strip_of_interest


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
    Read in arrays, process them, find residuals, and calculate KS test

    INPUTS:
    stripe_w_planet: integer which sets the strip with planets injected along the median angle
        (choices are [0,1,2,3,4])
    half_w_planet: the East/West half of the stripe_w_planet which contains the
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

    # files where planets are injected along the strip 0 of 4, along 129.68 deg E of N)
    # glob of file names of ADI frames of A block strip 0 of 4
    file_names_strip_0_of_4_planetsInStrip0_50pt32_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_0_of_4_planetsInStrip2_50pt32_deg/*.fits"))
    # glob of file names of ADI frames of D block strip 1 of 4, with planets aligned with strip 0 along 129.68 deg E of N
    file_names_strip_1_of_4_planetsInStrip0_50pt32_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_1_of_4_planetsInStrip2_50pt32_deg/*.fits"))
    # glob of file names of ADI frames of D block strip 2 of 4, with planets aligned with strip 0 along 129.68 deg E of N
    file_names_strip_2_of_4_planetsInStrip0_50pt32_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_2_of_4_planetsInStrip2_50pt32_deg/*.fits"))
    # glob of file names of ADI frames of D block strip 3 of 4, with planets aligned with strip 0 along 129.68 deg E of N
    file_names_strip_3_of_4_planetsInStrip0_50pt32_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_3_of_4_planetsInStrip2_50pt32_deg/*.fits"))
    # glob of file names of ADI frames of D block strip 4 of 4, with planets aligned with strip 0 along 129.68 deg E of N
    file_names_strip_4_of_4_planetsInStrip0_50pt32_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_4_of_4_planetsInStrip2_50pt32_deg/*.fits"))
    # globs of files with planets aligned with strip 0 again, but along opposite
    # azimuth of 230.32 deg E of N
    file_names_strip_0_of_4_planetsInStrip0_230pt32_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_0_of_4_planetsInStrip2_230pt32_deg/*.fits"))
    file_names_strip_1_of_4_planetsInStrip0_230pt32_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_1_of_4_planetsInStrip2_230pt32_deg/*.fits"))
    file_names_strip_2_of_4_planetsInStrip0_230pt32_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_2_of_4_planetsInStrip2_230pt32_deg/*.fits"))
    file_names_strip_3_of_4_planetsInStrip0_230pt32_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_3_of_4_planetsInStrip2_230pt32_deg/*.fits"))
    file_names_strip_4_of_4_planetsInStrip0_230pt32_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"pa_corrxn_jobs_strip_4_of_4_planetsInStrip2_230pt32_deg/*.fits"))

    # glob of file names of ADI frames with planets in strip 1 of 4, along 70.782 deg E of N
    #file_names_strip_0_of_4_planetsInStrip1_109pt218_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"__/*.fits"))

    # glob of file names of ADI frames with planets in strip 1 of 4, along 250.782 deg E of N
    #file_names_strip_0_of_4_planetsInStrip1_250pt782_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"__/*.fits"))

    # glob of file names of ADI frames with planets in strip 1 of 4, along 76.57 deg E of N
    #file_names_strip_0_of_4_planetsInStrip1_109pt218_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"__/*.fits"))

    # glob of file names of ADI frames with planets in strip 1 of 4, along 250.782 deg E of N
    #file_names_strip_0_of_4_planetsInStrip1_250pt782_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"__/*.fits"))

    # glob of file names of ADI frames with planets in strip 1 of 4, along 83.37 deg E of N
    #file_names_strip_0_of_4_planetsInStrip1_109pt218_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"__/*.fits"))

    # glob of file names of ADI frames with planets in strip 1 of 4, along 263.37 deg E of N
    #file_names_strip_0_of_4_planetsInStrip1_250pt782_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"__/*.fits"))

    # glob of file names of ADI frames with planets in strip 1 of 4, along 90.04 deg E of N
    #file_names_strip_0_of_4_planetsInStrip1_109pt218_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"__/*.fits"))

    # glob of file names of ADI frames with planets in strip 1 of 4, along 270.04 deg E of N
    #file_names_strip_0_of_4_planetsInStrip1_250pt782_deg = list(glob.glob(stem_adi_frames_lambda_over_B+"__/*.fits"))


    # choose the arrays to use in the analysis
    if (stripe_w_planet == 0):
        # frames with planet along the E: East of North; i.e., 0<PA<180
        file_names_strip_0_of_4_E = file_names_strip_0_of_4_planetsInStrip0_129pt68_deg
        file_names_strip_1_of_4_E = file_names_strip_1_of_4_planetsInStrip0_129pt68_deg
        file_names_strip_2_of_4_E = file_names_strip_2_of_4_planetsInStrip0_129pt68_deg
        file_names_strip_3_of_4_E = file_names_strip_3_of_4_planetsInStrip0_129pt68_deg
        file_names_strip_4_of_4_E = file_names_strip_4_of_4_planetsInStrip0_129pt68_deg
        # frames with planet along the W: West of North; i.e., 180<PA<360
        file_names_strip_0_of_4_W = file_names_strip_0_of_4_planetsInStrip0_309pt68_deg
        file_names_strip_1_of_4_W = file_names_strip_1_of_4_planetsInStrip0_309pt68_deg
        file_names_strip_2_of_4_W = file_names_strip_2_of_4_planetsInStrip0_309pt68_deg
        file_names_strip_3_of_4_W = file_names_strip_3_of_4_planetsInStrip0_309pt68_deg
        file_names_strip_4_of_4_W = file_names_strip_4_of_4_planetsInStrip0_309pt68_deg
        # for differentiating plot file names
        plot_string = "stripe_w_planet_0_"
        # name of the plot for the publication outside the for-loop below
        lambda_over_B_pub_plot_filename_suffix = plot_string + "pub_plot.pdf"
    elif (stripe_w_planet == 1):
        # E: East of North; i.e., 0<PA<180
        file_names_strip_0_of_4_E = file_names_strip_0_of_4_planetsInStrip1_109pt218_deg
        file_names_strip_1_of_4_E = file_names_strip_1_of_4_planetsInStrip1_109pt218_deg
        file_names_strip_2_of_4_E = file_names_strip_2_of_4_planetsInStrip1_109pt218_deg
        file_names_strip_3_of_4_E = file_names_strip_3_of_4_planetsInStrip1_109pt218_deg
        file_names_strip_4_of_4_E = file_names_strip_4_of_4_planetsInStrip1_109pt218_deg
        # W: West of North; i.e., 180<PA<360
        file_names_strip_0_of_4_W = file_names_strip_0_of_4_planetsInStrip1_289pt218_deg
        file_names_strip_1_of_4_W = file_names_strip_1_of_4_planetsInStrip1_289pt218_deg
        file_names_strip_2_of_4_W = file_names_strip_2_of_4_planetsInStrip1_289pt218_deg
        file_names_strip_3_of_4_W = file_names_strip_3_of_4_planetsInStrip1_289pt218_deg
        file_names_strip_4_of_4_W = file_names_strip_4_of_4_planetsInStrip1_289pt218_deg
        # for differentiating plot file names
        plot_string = "stripe_w_planet_1_"
        # name of the plot for the publication outside the for-loop below
        lambda_over_B_pub_plot_filename_suffix = plot_string + "pub_plot.pdf"
    elif (stripe_w_planet == 2):
        # E: East of North; i.e., 0<PA<180
        file_names_strip_0_of_4_E = file_names_strip_0_of_4_planetsInStrip2_103pt43_deg
        file_names_strip_1_of_4_E = file_names_strip_1_of_4_planetsInStrip2_103pt43_deg
        file_names_strip_2_of_4_E = file_names_strip_2_of_4_planetsInStrip2_103pt43_deg
        file_names_strip_3_of_4_E = file_names_strip_3_of_4_planetsInStrip2_103pt43_deg
        file_names_strip_4_of_4_E = file_names_strip_4_of_4_planetsInStrip2_103pt43_deg
        # W: West of North; i.e., 180<PA<360
        file_names_strip_0_of_4_W = file_names_strip_0_of_4_planetsInStrip2_283pt43_deg
        file_names_strip_1_of_4_W = file_names_strip_1_of_4_planetsInStrip2_283pt43_deg
        file_names_strip_2_of_4_W = file_names_strip_2_of_4_planetsInStrip2_283pt43_deg
        file_names_strip_3_of_4_W = file_names_strip_3_of_4_planetsInStrip2_283pt43_deg
        file_names_strip_4_of_4_W = file_names_strip_4_of_4_planetsInStrip2_283pt43_deg
        # for differentiating plot file names
        plot_string = "stripe_w_planet_2_"
        # name of the plot for the publication outside the for-loop below
        lambda_over_B_pub_plot_filename_suffix = plot_string + "pub_plot.pdf"
    elif (stripe_w_planet == 3):
        # E: East of North; i.e., 0<PA<180
        file_names_strip_0_of_4_E = file_names_strip_0_of_4_planetsInStrip3_96pt63_deg
        file_names_strip_1_of_4_E = file_names_strip_1_of_4_planetsInStrip3_96pt63_deg
        file_names_strip_2_of_4_E = file_names_strip_2_of_4_planetsInStrip3_96pt63_deg
        file_names_strip_3_of_4_E = file_names_strip_3_of_4_planetsInStrip3_96pt63_deg
        file_names_strip_4_of_4_E = file_names_strip_4_of_4_planetsInStrip3_96pt63_deg
        # W: West of North; i.e., 180<PA<360
        file_names_strip_0_of_4_W = file_names_strip_0_of_4_planetsInStrip3_276pt63_deg
        file_names_strip_1_of_4_W = file_names_strip_1_of_4_planetsInStrip3_276pt63_deg
        file_names_strip_2_of_4_W = file_names_strip_2_of_4_planetsInStrip3_276pt63_deg
        file_names_strip_3_of_4_W = file_names_strip_3_of_4_planetsInStrip3_276pt63_deg
        file_names_strip_4_of_4_W = file_names_strip_4_of_4_planetsInStrip3_276pt63_deg
        # for differentiating plot file names
        plot_string = "stripe_w_planet_3_"
        # name of the plot for the publication outside the for-loop below
        lambda_over_B_pub_plot_filename_suffix = plot_string + "pub_plot.pdf"
    elif (stripe_w_planet == 4):
        # E: East of North; i.e., 0<PA<180
        file_names_strip_0_of_4_E = file_names_strip_0_of_4_planetsInStrip4_89pt96_deg
        file_names_strip_1_of_4_E = file_names_strip_1_of_4_planetsInStrip4_89pt96_deg
        file_names_strip_2_of_4_E = file_names_strip_2_of_4_planetsInStrip4_89pt96_deg
        file_names_strip_3_of_4_E = file_names_strip_3_of_4_planetsInStrip4_89pt96_deg
        file_names_strip_4_of_4_E = file_names_strip_4_of_4_planetsInStrip4_89pt96_deg
        # W: West of North; i.e., 180<PA<360
        file_names_strip_0_of_4_W = file_names_strip_0_of_4_planetsInStrip4_269pt96_deg
        file_names_strip_1_of_4_W = file_names_strip_1_of_4_planetsInStrip4_269pt96_deg
        file_names_strip_2_of_4_W = file_names_strip_2_of_4_planetsInStrip4_269pt96_deg
        file_names_strip_3_of_4_W = file_names_strip_3_of_4_planetsInStrip4_269pt96_deg
        file_names_strip_4_of_4_W = file_names_strip_4_of_4_planetsInStrip4_269pt96_deg
        # for differentiating plot file names
        plot_string = "stripe_w_planet_4_"
        # name of the plot for the publication outside the for-loop below
        lambda_over_B_pub_plot_filename_suffix = plot_string + "pub_plot.pdf"
    else:
        print("Don't know which lists of file names to use in the analysis!")

    # initialize DataFrame to hold KS test info
    col_names = ["dist_asec",
                "comp_ampl",
                "id_strip_w_planet",
                "id_opposite_to_half_w_planet",
                "D_xsec_strip_w_planets_rel_to_same_half_strip_wo_planet",
                "D_xsec_strip_w_planets_rel_to_other_half_same_strip_with_planet",
                "D_xsec_strip_w_planets_rel_to_strip_0_E",
                "D_xsec_strip_w_planets_rel_to_strip_1_E",
                "D_xsec_strip_w_planets_rel_to_strip_2_E",
                "D_xsec_strip_w_planets_rel_to_strip_3_E",
                "D_xsec_strip_w_planets_rel_to_strip_4_E",
                "D_xsec_strip_w_planets_rel_to_strip_0_W",
                "D_xsec_strip_w_planets_rel_to_strip_1_W",
                "D_xsec_strip_w_planets_rel_to_strip_2_W",
                "D_xsec_strip_w_planets_rel_to_strip_3_W",
                "D_xsec_strip_w_planets_rel_to_strip_4_W",
                "val_xsec_crit_strip_w_planets_rel_to_strip_0_E",
                "val_xsec_crit_strip_w_planets_rel_to_strip_1_E",
                "val_xsec_crit_strip_w_planets_rel_to_strip_2_E",
                "val_xsec_crit_strip_w_planets_rel_to_strip_3_E",
                "val_xsec_crit_strip_w_planets_rel_to_strip_4_E",
                "val_xsec_crit_strip_w_planets_rel_to_strip_0_W",
                "val_xsec_crit_strip_w_planets_rel_to_strip_1_W",
                "val_xsec_crit_strip_w_planets_rel_to_strip_2_W",
                "val_xsec_crit_strip_w_planets_rel_to_strip_3_W",
                "val_xsec_crit_strip_w_planets_rel_to_strip_4_W"]
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
    comp_ampl_array = np.array([0.005,0.01,0.02,0.03,0.05,0.1])
    dist_fwhm_array = np.array([0.1,0.4,0.7,1.,1.3,1.7,2.0,2.3,2.6,2.9,3.2,4.,5.])
    fwhm_pix = 9.728 # FWHM for 4.05um/8.25m, in pixels
    dist_pix_array = np.multiply(dist_fwhm_array,fwhm_pix)
    dist_asec_array = np.multiply(dist_pix_array,float(config["instrum_params"]["LMIR_PS"]))

    # read in the baseline images with no planets
    image_baseline_stripe_0 = fits.getdata(file_name_strip_0_of_4_baseline_no_planet,0,header=False)
    baseline_processed_stripe_0 = shave_and_rotate(image_baseline_stripe_0,angle=50.32)
    #image_baseline_stripe_1 = fits.getdata(file_name_strip_1_of_4_baseline_no_planet,0,header=False)
    #baseline_processed_stripe_1 = shave_and_rotate(image_baseline_stripe_1,angle=19.218)
    #image_baseline_stripe_2 = fits.getdata(file_name_strip_2_of_4_baseline_no_planet,0,header=False)
    #baseline_processed_stripe_2 = shave_and_rotate(image_baseline_stripe_2,angle=13.43)
    #image_baseline_stripe_3 = fits.getdata(file_name_strip_3_of_4_baseline_no_planet,0,header=False)
    #baseline_processed_stripe_3 = shave_and_rotate(image_baseline_stripe_3,angle=6.63)
    #image_baseline_stripe_4 = fits.getdata(file_name_strip_4_of_4_baseline_no_planet,0,header=False)
    #baseline_processed_stripe_4 = shave_and_rotate(image_baseline_stripe_4,angle=-0.04)

    # loop over all fake planet east-west pairs distinguished by amplitude and radius
    for comp_ampl_num in range(0,len(comp_ampl_array)):
        for dist_asec_num in range(0,len(dist_asec_array)):

            comp_ampl = comp_ampl_array[comp_ampl_num]
            dist_asec = dist_asec_array[dist_asec_num]

            # name of the FYI plot to write
            new_filename = plot_string + \
                            "lambda_over_B_comp_ampl_" + str(comp_ampl) + \
                            "_dist_asec_" + str(dist_asec) + ".png"

            print("---------------------------------------------------")
            print("Doing KS test for comp_ampl " + str(comp_ampl) + \
                                " and dist_asec " + str(dist_asec))

            ## pluck out the interesting file names
            # frames with planets along eastern arm of strip
            file_name_strip_0_of_4_E = pluck_interesting_file_name(file_names_strip_0_of_4_E,
                                                                 comp_ampl_pass=comp_ampl,
                                                                 dist_asec_pass=dist_asec)
            file_name_strip_1_of_4_E = pluck_interesting_file_name(file_names_strip_1_of_4_E,
                                                                 comp_ampl_pass=comp_ampl,
                                                                 dist_asec_pass=dist_asec)
            file_name_strip_2_of_4_E = pluck_interesting_file_name(file_names_strip_2_of_4_E,
                                                                 comp_ampl_pass=comp_ampl,
                                                                 dist_asec_pass=dist_asec)
            file_name_strip_3_of_4_E = pluck_interesting_file_name(file_names_strip_3_of_4_E,
                                                                 comp_ampl_pass=comp_ampl,
                                                                 dist_asec_pass=dist_asec)
            file_name_strip_4_of_4_E = pluck_interesting_file_name(file_names_strip_4_of_4_E,
                                                                comp_ampl_pass=comp_ampl,
                                                                dist_asec_pass=dist_asec)
            # frames with planets along western arm of strip
            file_name_strip_0_of_4_W = pluck_interesting_file_name(file_names_strip_0_of_4_W,
                                                                 comp_ampl_pass=comp_ampl,
                                                                 dist_asec_pass=dist_asec)
            file_name_strip_1_of_4_W = pluck_interesting_file_name(file_names_strip_1_of_4_W,
                                                                 comp_ampl_pass=comp_ampl,
                                                                 dist_asec_pass=dist_asec)
            file_name_strip_2_of_4_W = pluck_interesting_file_name(file_names_strip_2_of_4_W,
                                                                 comp_ampl_pass=comp_ampl,
                                                                 dist_asec_pass=dist_asec)
            file_name_strip_3_of_4_W = pluck_interesting_file_name(file_names_strip_3_of_4_W,
                                                                 comp_ampl_pass=comp_ampl,
                                                                 dist_asec_pass=dist_asec)
            file_name_strip_4_of_4_W = pluck_interesting_file_name(file_names_strip_4_of_4_W,
                                                                comp_ampl_pass=comp_ampl,
                                                                dist_asec_pass=dist_asec)

            ## read in and process the images with planets
            # frames with planets along eastern arm of strip
            image_stripe_0_E = fits.getdata(file_name_strip_0_of_4_E,0,header=False)
            img_processed_stripe_0_E = shave_and_rotate(image_stripe_0_E,angle=39.68)
            image_stripe_1_E = fits.getdata(file_name_strip_1_of_4_E,0,header=False)
            img_processed_stripe_1_E = shave_and_rotate(image_stripe_1_E,angle=19.218)
            image_stripe_2_E = fits.getdata(file_name_strip_2_of_4_E,0,header=False)
            img_processed_stripe_2_E = shave_and_rotate(image_stripe_2_E,angle=13.43)
            image_stripe_3_E = fits.getdata(file_name_strip_3_of_4_E,0,header=False)
            img_processed_stripe_3_E = shave_and_rotate(image_stripe_3_E,angle=6.63)
            image_stripe_4_E = fits.getdata(file_name_strip_4_of_4_E,0,header=False)
            img_processed_stripe_4_E = shave_and_rotate(image_stripe_4_E,angle=-0.04)
            # frames with planets along western arm of strip
            image_stripe_0_W = fits.getdata(file_name_strip_0_of_4_W,0,header=False)
            img_processed_stripe_0_W = shave_and_rotate(image_stripe_0_W,angle=39.68)
            image_stripe_1_W = fits.getdata(file_name_strip_1_of_4_W,0,header=False)
            img_processed_stripe_1_W = shave_and_rotate(image_stripe_1_W,angle=19.218)
            image_stripe_2_W = fits.getdata(file_name_strip_2_of_4_W,0,header=False)
            img_processed_stripe_2_W = shave_and_rotate(image_stripe_2_W,angle=13.43)
            image_stripe_3_W = fits.getdata(file_name_strip_3_of_4_W,0,header=False)
            img_processed_stripe_3_W = shave_and_rotate(image_stripe_3_W,angle=6.63)
            image_stripe_4_W = fits.getdata(file_name_strip_4_of_4_W,0,header=False)
            img_processed_stripe_4_W = shave_and_rotate(image_stripe_4_W,angle=-0.04)
            import ipdb; ipdb.set_trace()
            # find the cross-sections and marginalizations

            # initialize dictionaries
            marginalization_dict = {}
            cross_sec_dict = {}

            ## marginalizations
            ## (AS OF 2020 JUNE 2 THESE ARE ACTUALLY IGNORED, BECAUSE I DID NOT
            ## FIND THEM TO BE VERY INDICATIVE; NOTE I HAVE NOT TAKEN THE
            ## TIME TO SEPARATE BASELINES INTO E, W HALVES --E.S.)
            # baseline frames without planets
            marginalization_dict["baseline_strip_0"] = np.sum(baseline_processed_stripe_0,axis=0)
            marginalization_dict["baseline_strip_1"] = np.sum(baseline_processed_stripe_1,axis=0)
            marginalization_dict["baseline_strip_2"] = np.sum(baseline_processed_stripe_2,axis=0)
            marginalization_dict["baseline_strip_3"] = np.sum(baseline_processed_stripe_3,axis=0)
            marginalization_dict["baseline_strip_4"] = np.sum(baseline_processed_stripe_4,axis=0)
            # frames with planets in eastern or western arm
            # (NOTE THERE IS NO FLIPPING OF 'EAST' HALVES HERE, UNLESS MARGINALIZATION
            # PROVES USEFUL LATER ON)
            marginalization_dict["strip_0_E"] = np.sum(img_processed_stripe_0_E,axis=0)
            marginalization_dict["strip_1_E"] = np.sum(img_processed_stripe_1_E,axis=0)
            marginalization_dict["strip_2_E"] = np.sum(img_processed_stripe_2_E,axis=0)
            marginalization_dict["strip_3_E"] = np.sum(img_processed_stripe_3_E,axis=0)
            marginalization_dict["strip_4_E"] = np.sum(img_processed_stripe_4_E,axis=0)
            marginalization_dict["strip_0_W"] = np.sum(img_processed_stripe_0_W,axis=0)
            marginalization_dict["strip_1_W"] = np.sum(img_processed_stripe_1_W,axis=0)
            marginalization_dict["strip_2_W"] = np.sum(img_processed_stripe_2_W,axis=0)
            marginalization_dict["strip_3_W"] = np.sum(img_processed_stripe_3_W,axis=0)
            marginalization_dict["strip_4_W"] = np.sum(img_processed_stripe_4_W,axis=0)

            ## cross-sections
            # baseline frames without planets: eastern halves and western halves
            # set down convention that host star is on the left side (i.e., 'east'
            # strips have to be flipped)
            cross_sec_dict["baseline_strip_0_E"] = np.flip(baseline_processed_stripe_0, axis=1)[int(0.5*np.shape(baseline_processed_stripe_0)[0]),int(0.5*np.shape(baseline_processed_stripe_0)[1]):]
            cross_sec_dict["baseline_strip_1_E"] = np.flip(baseline_processed_stripe_1, axis=1)[int(0.5*np.shape(baseline_processed_stripe_1)[0]),int(0.5*np.shape(baseline_processed_stripe_1)[1]):]
            cross_sec_dict["baseline_strip_2_E"] = np.flip(baseline_processed_stripe_2, axis=1)[int(0.5*np.shape(baseline_processed_stripe_2)[0]),int(0.5*np.shape(baseline_processed_stripe_2)[1]):]
            cross_sec_dict["baseline_strip_3_E"] = np.flip(baseline_processed_stripe_3, axis=1)[int(0.5*np.shape(baseline_processed_stripe_3)[0]),int(0.5*np.shape(baseline_processed_stripe_3)[1]):]
            cross_sec_dict["baseline_strip_4_E"] = np.flip(baseline_processed_stripe_4, axis=1)[int(0.5*np.shape(baseline_processed_stripe_4)[0]),int(0.5*np.shape(baseline_processed_stripe_4)[1]):]
            cross_sec_dict["baseline_strip_0_W"] = baseline_processed_stripe_0[int(0.5*np.shape(baseline_processed_stripe_0)[0]),int(0.5*np.shape(baseline_processed_stripe_0)[1]):]
            cross_sec_dict["baseline_strip_1_W"] = baseline_processed_stripe_1[int(0.5*np.shape(baseline_processed_stripe_1)[0]),int(0.5*np.shape(baseline_processed_stripe_1)[1]):]
            cross_sec_dict["baseline_strip_2_W"] = baseline_processed_stripe_2[int(0.5*np.shape(baseline_processed_stripe_2)[0]),int(0.5*np.shape(baseline_processed_stripe_2)[1]):]
            cross_sec_dict["baseline_strip_3_W"] = baseline_processed_stripe_3[int(0.5*np.shape(baseline_processed_stripe_3)[0]),int(0.5*np.shape(baseline_processed_stripe_3)[1]):]
            cross_sec_dict["baseline_strip_4_W"] = baseline_processed_stripe_4[int(0.5*np.shape(baseline_processed_stripe_4)[0]),int(0.5*np.shape(baseline_processed_stripe_4)[1]):]
            # frames with planets in eastern or western arm
            cross_sec_dict["strip_0_E"] = np.flip(img_processed_stripe_0_E, axis=1)[int(0.5*np.shape(img_processed_stripe_0_E)[0]),int(0.5*np.shape(img_processed_stripe_0_E)[1]):]
            cross_sec_dict["strip_1_E"] = np.flip(img_processed_stripe_1_E, axis=1)[int(0.5*np.shape(img_processed_stripe_1_E)[0]),int(0.5*np.shape(img_processed_stripe_1_E)[1]):]
            cross_sec_dict["strip_2_E"] = np.flip(img_processed_stripe_2_E, axis=1)[int(0.5*np.shape(img_processed_stripe_2_E)[0]),int(0.5*np.shape(img_processed_stripe_2_E)[1]):]
            cross_sec_dict["strip_3_E"] = np.flip(img_processed_stripe_3_E, axis=1)[int(0.5*np.shape(img_processed_stripe_3_E)[0]),int(0.5*np.shape(img_processed_stripe_3_E)[1]):]
            cross_sec_dict["strip_4_E"] = np.flip(img_processed_stripe_4_E, axis=1)[int(0.5*np.shape(img_processed_stripe_4_E)[0]),int(0.5*np.shape(img_processed_stripe_4_E)[1]):]
            cross_sec_dict["strip_0_W"] = img_processed_stripe_0_W[int(0.5*np.shape(img_processed_stripe_0_W)[0]),int(0.5*np.shape(img_processed_stripe_0_W)[1]):]
            cross_sec_dict["strip_1_W"] = img_processed_stripe_1_W[int(0.5*np.shape(img_processed_stripe_1_W)[0]),int(0.5*np.shape(img_processed_stripe_1_W)[1]):]
            cross_sec_dict["strip_2_W"] = img_processed_stripe_2_W[int(0.5*np.shape(img_processed_stripe_2_W)[0]),int(0.5*np.shape(img_processed_stripe_2_W)[1]):]
            cross_sec_dict["strip_3_W"] = img_processed_stripe_3_W[int(0.5*np.shape(img_processed_stripe_3_W)[0]),int(0.5*np.shape(img_processed_stripe_3_W)[1]):]
            cross_sec_dict["strip_4_W"] = img_processed_stripe_4_W[int(0.5*np.shape(img_processed_stripe_4_W)[0]),int(0.5*np.shape(img_processed_stripe_4_W)[1]):]
            import ipdb; ipdb.set_trace()
            if (stripe_w_planet == 0):
                if (half_w_planet == "E"):
                    image_injected_planet = np.flip(img_processed_stripe_0_E, axis=1) # note flipping of 'east' half
                    cross_sec_injected_planet = cross_sec_dict["strip_0_E"] # already flipped
                    cross_sec_baseline = cross_sec_dict["baseline_strip_0_E"] # already flipped
                    # we need to check for azimuthal variations that are not just
                    # due to phase changes between one stripe and another; so,
                    # here is a string for tagging the stats relevant to the opposite
                    # half of the stripe which has the planet injected
                    string_w_planet_indicator = "strip_0_E"
                    string_opposite_indicator = "strip_0_W"
                elif (half_w_planet == "W"):
                    image_injected_planet = img_processed_stripe_0_W
                    cross_sec_injected_planet = cross_sec_dict["strip_0_W"]
                    cross_sec_baseline = cross_sec_dict["baseline_strip_0_W"]
                    string_w_planet_indicator = "strip_0_W"
                    string_opposite_indicator = "strip_0_E"
                marginalization_baseline = marginalization_dict["baseline_strip_0"] # note I'm not bothering with flipping since marginalization is not being used as of 2020 June 2
                marginalization_injected_planet_E = marginalization_dict["strip_0_E"]
                marginalization_injected_planet_W = marginalization_dict["strip_0_W"]
            elif (stripe_w_planet == 1):
                if (half_w_planet == "E"):
                    image_injected_planet = np.flip(img_processed_stripe_1_E, axis=1) # note flipping of 'east' half
                    cross_sec_injected_planet = cross_sec_dict["strip_1_E"] # already flipped
                    cross_sec_baseline = cross_sec_dict["baseline_strip_1_E"] # already flipped
                    string_w_planet_indicator = "strip_1_E"
                    string_opposite_indicator = "strip_1_W"
                elif (half_w_planet == "W"):
                    image_injected_planet = img_processed_stripe_1_W
                    cross_sec_injected_planet = cross_sec_dict["strip_1_W"]
                    cross_sec_baseline = cross_sec_dict["baseline_strip_1_W"]
                    string_w_planet_indicator = "strip_1_W"
                    string_opposite_indicator = "strip_1_E"
                marginalization_baseline = marginalization_dict["baseline_strip_1"] # note I'm not bothering with flipping since marginalization is not being used as of 2020 June 2
                marginalization_injected_planet_E = marginalization_dict["strip_1_E"]
                marginalization_injected_planet_W = marginalization_dict["strip_1_W"]
            elif (stripe_w_planet == 2):
                if (half_w_planet == "E"):
                    image_injected_planet = np.flip(img_processed_stripe_2_E, axis=1) # note flipping of 'east' half
                    cross_sec_injected_planet = cross_sec_dict["strip_2_E"] # already flipped
                    cross_sec_baseline = cross_sec_dict["baseline_strip_2_E"] # already flipped
                    string_w_planet_indicator = "strip_2_E"
                    string_opposite_indicator = "strip_2_W"
                elif (half_w_planet == "W"):
                    image_injected_planet = img_processed_stripe_2_W
                    cross_sec_injected_planet = cross_sec_dict["strip_2_W"]
                    cross_sec_baseline = cross_sec_dict["baseline_strip_2_W"]
                    string_w_planet_indicator = "strip_2_W"
                    string_opposite_indicator = "strip_2_E"
                marginalization_baseline = marginalization_dict["baseline_strip_2"] # note I'm not bothering with flipping since marginalization is not being used as of 2020 June 2
                marginalization_injected_planet_E = marginalization_dict["strip_2_E"]
                marginalization_injected_planet_W = marginalization_dict["strip_2_W"]
            elif (stripe_w_planet == 3):
                if (half_w_planet == "E"):
                    image_injected_planet = np.flip(img_processed_stripe_3_E, axis=1) # note flipping of 'east' half
                    cross_sec_injected_planet = cross_sec_dict["strip_3_E"] # already flipped
                    cross_sec_baseline = cross_sec_dict["baseline_strip_3_E"] # already flipped
                    string_w_planet_indicator = "strip_3_E"
                    string_opposite_indicator = "strip_3_W"
                elif (half_w_planet == "W"):
                    image_injected_planet = img_processed_stripe_3_W
                    cross_sec_injected_planet = cross_sec_dict["strip_3_W"]
                    cross_sec_baseline = cross_sec_dict["baseline_strip_3_W"]
                    string_w_planet_indicator = "strip_3_W"
                    string_opposite_indicator = "strip_3_E"
                marginalization_baseline = marginalization_dict["baseline_strip_3"] # note I'm not bothering with flipping since marginalization is not being used as of 2323 June 2
                marginalization_injected_planet_E = marginalization_dict["strip_3_E"]
                marginalization_injected_planet_W = marginalization_dict["strip_3_W"]
            elif (stripe_w_planet == 4):
                if (half_w_planet == "E"):
                    image_injected_planet = np.flip(img_processed_stripe_4_E, axis=1) # note flipping of 'east' half
                    cross_sec_injected_planet = cross_sec_dict["strip_4_E"] # already flipped
                    cross_sec_baseline = cross_sec_dict["baseline_strip_4_E"] # already flipped
                    string_w_planet_indicator = "strip_4_E"
                    string_opposite_indicator = "strip_4_W"
                elif (half_w_planet == "W"):
                    image_injected_planet = img_processed_stripe_4_W
                    cross_sec_injected_planet = cross_sec_dict["strip_4_W"]
                    cross_sec_baseline = cross_sec_dict["baseline_strip_4_W"]
                    string_w_planet_indicator = "strip_4_W"
                    string_opposite_indicator = "strip_4_E"
                marginalization_baseline = marginalization_dict["baseline_strip_4"] # note I'm not bothering with flipping since marginalization is not being used as of 2020 June 2
                marginalization_injected_planet_E = marginalization_dict["strip_4_E"]
                marginalization_injected_planet_W = marginalization_dict["strip_4_W"]
            else:
                print("No strip with planet specified!")
            import ipdb; ipdb.set_trace()
            ## calculate relevant quantities, put them into dataframe
            # KS statistic from cross-sections
            # baseline: comparison with same strip but without any planet at all
            strip_baseline_ks_cross_sec = do_KS(cross_sec_injected_planet,cross_sec_baseline)
            # opposite half: comparison with the *opposite* side of the strip
            # with the planet injected
            strip_opposite_ks_cross_sec = do_KS(cross_sec_injected_planet,cross_sec_dict[string_opposite_indicator])
            # others: different strips with planets along the same angle
            # (note one of these will be a repeat of strip_opposite_ks_cross_sec,
            # which will be removed further downstream in the pipeline)
            #cross_sec_dict[string_opposite_indicator] = np.multiply(np.nan,cross_sec_dict[string_opposite_indicator])
            strip_0_ks_cross_sec_E = do_KS(cross_sec_injected_planet,cross_sec_dict["strip_0_E"])
            strip_1_ks_cross_sec_E = do_KS(cross_sec_injected_planet,cross_sec_dict["strip_1_E"])
            strip_2_ks_cross_sec_E = do_KS(cross_sec_injected_planet,cross_sec_dict["strip_2_E"])
            strip_3_ks_cross_sec_E = do_KS(cross_sec_injected_planet,cross_sec_dict["strip_3_E"])
            strip_4_ks_cross_sec_E = do_KS(cross_sec_injected_planet,cross_sec_dict["strip_4_E"])
            strip_0_ks_cross_sec_W = do_KS(cross_sec_injected_planet,cross_sec_dict["strip_0_W"])
            strip_1_ks_cross_sec_W = do_KS(cross_sec_injected_planet,cross_sec_dict["strip_1_W"])
            strip_2_ks_cross_sec_W = do_KS(cross_sec_injected_planet,cross_sec_dict["strip_2_W"])
            strip_3_ks_cross_sec_W = do_KS(cross_sec_injected_planet,cross_sec_dict["strip_3_W"])
            strip_4_ks_cross_sec_W = do_KS(cross_sec_injected_planet,cross_sec_dict["strip_4_W"])
            # KS statistic from marginalizations
            strip_0_ks_marg_E = do_KS(marginalization_injected_planet_E,marginalization_dict["strip_0_E"])
            strip_1_ks_marg_E = do_KS(marginalization_injected_planet_E,marginalization_dict["strip_1_E"])
            strip_2_ks_marg_E = do_KS(marginalization_injected_planet_E,marginalization_dict["strip_2_E"])
            strip_3_ks_marg_E = do_KS(marginalization_injected_planet_E,marginalization_dict["strip_3_E"])
            strip_4_ks_marg_E = do_KS(marginalization_injected_planet_E,marginalization_dict["strip_4_E"])
            strip_0_ks_marg_W = do_KS(marginalization_injected_planet_W,marginalization_dict["strip_0_W"])
            strip_1_ks_marg_W = do_KS(marginalization_injected_planet_W,marginalization_dict["strip_1_W"])
            strip_2_ks_marg_W = do_KS(marginalization_injected_planet_W,marginalization_dict["strip_2_W"])
            strip_3_ks_marg_W = do_KS(marginalization_injected_planet_W,marginalization_dict["strip_3_W"])
            strip_4_ks_marg_W = do_KS(marginalization_injected_planet_W,marginalization_dict["strip_4_W"])

            # put stats into a dictionary; note there are entries corresponding to
            # injected planets on the E and W arms of the strips
            my_dic = {"dist_asec": dist_asec,
                    "comp_ampl": comp_ampl,
                    "id_strip_w_planet": string_w_planet_indicator,
                    "id_opposite_to_half_w_planet": string_opposite_indicator,
                    "D_xsec_strip_w_planets_rel_to_same_half_strip_wo_planet": strip_baseline_ks_cross_sec[0],
                    "D_xsec_strip_w_planets_rel_to_other_half_same_strip_with_planet": strip_opposite_ks_cross_sec[0],
                    "D_xsec_strip_w_planets_rel_to_strip_0_E": strip_0_ks_cross_sec_E[0],
                    "D_xsec_strip_w_planets_rel_to_strip_1_E": strip_1_ks_cross_sec_E[0],
                    "D_xsec_strip_w_planets_rel_to_strip_2_E": strip_2_ks_cross_sec_E[0],
                    "D_xsec_strip_w_planets_rel_to_strip_3_E": strip_3_ks_cross_sec_E[0],
                    "D_xsec_strip_w_planets_rel_to_strip_4_E": strip_4_ks_cross_sec_E[0],
                    "D_xsec_strip_w_planets_rel_to_strip_0_W": strip_0_ks_cross_sec_W[0],
                    "D_xsec_strip_w_planets_rel_to_strip_1_W": strip_1_ks_cross_sec_W[0],
                    "D_xsec_strip_w_planets_rel_to_strip_2_W": strip_2_ks_cross_sec_W[0],
                    "D_xsec_strip_w_planets_rel_to_strip_3_W": strip_3_ks_cross_sec_W[0],
                    "D_xsec_strip_w_planets_rel_to_strip_4_W": strip_4_ks_cross_sec_W[0],
                    "val_xsec_crit_strip_w_planets_rel_to_strip_0_E": strip_0_ks_cross_sec_E[1],
                    "val_xsec_crit_strip_w_planets_rel_to_strip_1_E": strip_1_ks_cross_sec_E[1],
                    "val_xsec_crit_strip_w_planets_rel_to_strip_2_E": strip_2_ks_cross_sec_E[1],
                    "val_xsec_crit_strip_w_planets_rel_to_strip_3_E": strip_3_ks_cross_sec_E[1],
                    "val_xsec_crit_strip_w_planets_rel_to_strip_4_E": strip_4_ks_cross_sec_E[1],
                    "val_xsec_crit_strip_w_planets_rel_to_strip_0_W": strip_0_ks_cross_sec_W[1],
                    "val_xsec_crit_strip_w_planets_rel_to_strip_1_W": strip_1_ks_cross_sec_W[1],
                    "val_xsec_crit_strip_w_planets_rel_to_strip_2_W": strip_2_ks_cross_sec_W[1],
                    "val_xsec_crit_strip_w_planets_rel_to_strip_3_W": strip_3_ks_cross_sec_W[1],
                    "val_xsec_crit_strip_w_planets_rel_to_strip_4_W": strip_4_ks_cross_sec_W[1]}

            ks_info_df.loc[len(ks_info_df)] = my_dic

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

            ##############################################
            ## BEGIN GIANT BLOCK OF CODE TO MAKE A PLOT

            # planet location in pixels to indicate in plot
            planet_loc_pix = np.divide(dist_asec,float(config["instrum_params"]["LMIR_PS"]))

            f, ((ax1, ax2, ax3, ax4, ax5, ax6), (ax7, ax8, ax9, ax10, ax11, ax12)) = plt.subplots(2, 6, figsize=(24, 16))

            # top row: 2D color plot and cross-sections
            # bottom row: marginalizations

            # top left: 2D color plot
            subplot1 = ax1.imshow(image_injected_planet, origin="lower", aspect="auto", vmin=-5000, vmax=5000)
            ax1.axvline(x=0.5*np.shape(image_injected_planet)[0]-planet_loc_pix,
                        linestyle=":", color="k", linewidth=4, alpha=0.4)
            #plt.colorbar(subplot1)

            # plot cross-sections and their differences between different strips
            ax2.plot(cross_sec_dict["strip_0_E"], label="cross sec")
            ax2.plot(np.subtract(cross_sec_injected_planet,cross_sec_dict["strip_0_E"]), label="diff")
            ax2.axvline(x=0.5*np.shape(image_injected_planet)[0]-planet_loc_pix,
                linestyle=":", color="k", linewidth=4, alpha=0.4)

            ax2.legend()
            ax2.set_title("Cross-sec rel. to strip 0, E\nD = "
                          + str(np.round(strip_0_ks_cross_sec_E[0],4))
                          + ",\nval_crit = " + str(np.round(strip_0_ks_cross_sec_E[1],4))
                          + ",\np_val = " + str(np.round(strip_0_ks_cross_sec_E[2],4)))

            ax3.plot(cross_sec_dict["strip_1_E"], label="cross sec")
            ax3.plot(np.subtract(cross_sec_injected_planet,cross_sec_dict["strip_1_E"]), label="diff")
            ax3.axvline(x=0.5*np.shape(image_injected_planet)[0]-planet_loc_pix,
                linestyle=":", color="k", linewidth=4, alpha=0.4)
            ax3.legend()
            ax3.set_title("Cross-sec rel. to strip 1, E\nD = "
                          + str(np.round(strip_1_ks_cross_sec_E[0],4)) + ",\nval_crit = "
                          + str(np.round(strip_1_ks_cross_sec_E[1],4)) + ",\np_val = "
                          + str(np.round(strip_1_ks_cross_sec_E[2],4)))


            ax4.plot(cross_sec_dict["strip_2_E"], label="cross sec")
            ax4.plot(np.subtract(cross_sec_injected_planet,cross_sec_dict["strip_2_E"]), label="diff")
            ax4.axvline(x=0.5*np.shape(image_injected_planet)[0]-planet_loc_pix,
                linestyle=":", color="k", linewidth=4, alpha=0.4)
            ax4.legend()
            ax4.set_title("Cross-sec rel. to strip 2, E\nD = "
                          + str(np.round(strip_2_ks_cross_sec_E[0],4)) + ",\nval_crit = "
                          + str(np.round(strip_2_ks_cross_sec_E[1],4)) + ",\np_val = "
                          + str(np.round(strip_2_ks_cross_sec_E[2],4)))


            ax5.plot(cross_sec_dict["strip_3_E"], label="cross sec")
            ax5.plot(np.subtract(cross_sec_injected_planet,cross_sec_dict["strip_3_E"]), label="diff")
            ax5.axvline(x=0.5*np.shape(image_injected_planet)[0]-planet_loc_pix,
                linestyle=":", color="k", linewidth=4, alpha=0.4)
            ax5.legend()
            ax5.set_title("Cross-sec rel. to strip 3, E\nD = "
                          + str(np.round(strip_3_ks_cross_sec_E[0],4)) + ",\nval_crit = "
                          + str(np.round(strip_3_ks_cross_sec_E[1],4)) + ",\np_val = "
                          + str(np.round(strip_3_ks_cross_sec_E[2],4)))


            ax6.plot(cross_sec_dict["strip_4_E"], label="cross sec")
            ax6.plot(np.subtract(cross_sec_injected_planet,cross_sec_dict["strip_4_E"]), label="diff")
            ax6.axvline(x=0.5*np.shape(image_injected_planet)[0]-planet_loc_pix,
                linestyle=":", color="k", linewidth=4, alpha=0.4)
            ax6.legend()
            ax6.set_title("Cross-sec rel. to strip 4, E\nD = "
                          + str(np.round(strip_4_ks_cross_sec_E[0],4)) + ",\nval_crit = "
                          + str(np.round(strip_4_ks_cross_sec_E[1],4)) + ",\np_val = "
                          + str(np.round(strip_4_ks_cross_sec_E[2],4)))


            # bottom-left (ax7): blank

            # plot cross-sections and their differences between different strips
            ax8.plot(cross_sec_dict["strip_0_W"], label="cross sec")
            ax8.plot(np.subtract(cross_sec_injected_planet,cross_sec_dict["strip_0_W"]), label="diff")
            ax8.axvline(x=0.5*np.shape(image_injected_planet)[0]-planet_loc_pix,
                linestyle=":", color="k", linewidth=4, alpha=0.4)

            ax8.legend()
            ax8.set_title("Cross-sec rel. to strip 0, W\nD = "
                          + str(np.round(strip_0_ks_cross_sec_W[0],4))
                          + ",\nval_crit = " + str(np.round(strip_0_ks_cross_sec_W[1],4))
                          + ",\np_val = " + str(np.round(strip_0_ks_cross_sec_W[2],4)))

            ax9.plot(cross_sec_dict["strip_1_W"], label="cross sec")
            ax9.plot(np.subtract(cross_sec_injected_planet,cross_sec_dict["strip_1_W"]), label="diff")
            ax9.axvline(x=0.5*np.shape(image_injected_planet)[0]-planet_loc_pix,
                linestyle=":", color="k", linewidth=4, alpha=0.4)
            ax9.legend()
            ax9.set_title("Cross-sec rel. to strip 1, W\nD = "
                          + str(np.round(strip_1_ks_cross_sec_W[0],4)) + ",\nval_crit = "
                          + str(np.round(strip_1_ks_cross_sec_W[1],4)) + ",\np_val = "
                          + str(np.round(strip_1_ks_cross_sec_W[2],4)))


            ax10.plot(cross_sec_dict["strip_2_W"], label="cross sec")
            ax10.plot(np.subtract(cross_sec_injected_planet,cross_sec_dict["strip_2_W"]), label="diff")
            ax10.axvline(x=0.5*np.shape(image_injected_planet)[0]-planet_loc_pix,
                linestyle=":", color="k", linewidth=4, alpha=0.4)
            ax10.legend()
            ax10.set_title("Cross-sec rel. to strip 2, W\nD = "
                          + str(np.round(strip_2_ks_cross_sec_W[0],4)) + ",\nval_crit = "
                          + str(np.round(strip_2_ks_cross_sec_W[1],4)) + ",\np_val = "
                          + str(np.round(strip_2_ks_cross_sec_W[2],4)))


            ax11.plot(cross_sec_dict["strip_3_W"], label="cross sec")
            ax11.plot(np.subtract(cross_sec_injected_planet,cross_sec_dict["strip_3_W"]), label="diff")
            ax11.axvline(x=0.5*np.shape(image_injected_planet)[0]-planet_loc_pix,
                linestyle=":", color="k", linewidth=4, alpha=0.4)
            ax11.legend()
            ax11.set_title("Cross-sec rel. to strip 3, W\nD = "
                          + str(np.round(strip_3_ks_cross_sec_W[0],4)) + ",\nval_crit = "
                          + str(np.round(strip_3_ks_cross_sec_W[1],4)) + ",\np_val = "
                          + str(np.round(strip_3_ks_cross_sec_W[2],4)))


            ax12.plot(cross_sec_dict["strip_4_W"], label="cross sec")
            ax12.plot(np.subtract(cross_sec_injected_planet,cross_sec_dict["strip_4_W"]), label="diff")
            ax12.axvline(x=0.5*np.shape(image_injected_planet)[0]-planet_loc_pix,
                linestyle=":", color="k", linewidth=4, alpha=0.4)
            ax12.legend()
            ax12.set_title("Cross-sec rel. to strip 4, W\nD = "
                          + str(np.round(strip_4_ks_cross_sec_W[0],4)) + ",\nval_crit = "
                          + str(np.round(strip_4_ks_cross_sec_W[1],4)) + ",\np_val = "
                          + str(np.round(strip_4_ks_cross_sec_W[2],4)))

            #f.suptitle(plot_file_name_prefix + os.path.basename(file_name_array_choice[file_num]))
            #plt.tight_layout()
            plt.savefig(new_filename, dpi=150)
            plt.close()
            #plt.show()

            print("Saved " + new_filename)

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
