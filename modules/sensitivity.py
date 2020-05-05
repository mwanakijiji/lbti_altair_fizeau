import multiprocessing
import configparser
import glob
import time
import pickle
import math
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t
from astropy.io import fits
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from modules import *


# function to convert radial data to xy, where target is in south
# (north is up (i.e., +y), degrees are E of N (i.e., CCW of +y axis))

def convert_rad_xy(canvas_array, PS, rho_asec, theta_deg):
    '''
    INPUTS:
    canvas_array: the array will will plot data on (and which you
        probably want to be oversampled)
    PS: effective plate scale for the display area (i.e., asec/pixel)
    rho: array of radii (in asec)
    theta_deg: array of angles E of N (in degrees)
    '''

    # find the center of the dummy array; this will be the pivot point
    array_center = (int(0.5*np.shape(canvas_array)[0]),
                    int(0.5*np.shape(canvas_array)[1]))

    # find x, y offsets from center in units of display pixels
    offset_from_center_x = -np.divide(rho_asec,PS)*np.sin(np.multiply(theta_deg,np.pi/180.))
    offset_from_center_y = np.divide(rho_asec,PS)*np.cos(np.multiply(theta_deg,np.pi/180.))

    # find absolute coordinates
    y_absolute = np.add(array_center[0],offset_from_center_y)
    x_absolute = np.add(array_center[1],offset_from_center_x)

    return y_absolute, x_absolute

def one_d_modern_contrast():
    '''
    Produces a 1D contrast curve (for regime of lambda/D)
    corrected for small angles and FPF, TPF constraints

    INPUTS:
    sbar_and_r_asec_pass: pandas dataframe with
        ["contrast_lin"]: classical, linear empirical contrast, or the 's-bar'
                            (implied TPF=0.5 but the curve says nothing else)
        ["rad_asec"]: radius from host star in arcsec
    TPF_pass: the fixed true positive fraction (default 0.95)

    OUTPUTS:
    correction_factor: the factor by which to multiply the input 1-sigma contrast curve
    corrected_curve: the actual corrected '5-sigma' curve, found by using the correction factor
    '''

    # read in the empirical 5-sigma linearly-scaled contrast, without correction
    # the input curve, as is, represents the relative amplitude at which TPF=0.5, and nothing else


    ## ## PLACEHOLDER DATA
    original_contrast_curve = pd.read_csv("./notebooks_for_development/data/fake_contrast_curve.csv")
    ## ## FOR THE REAL THING, UNCOMMENT THE BELOW
    #sbar_and_r_asec_pass = original_contrast_curve

    # Set some initial constants

    # Note that radii are all in units of FWHM unless explicity
    # stated otherwise
    N_FP_tot = 0.01 # limit on total number of false positives in the entire dataset
    R_max = 20 # maximum radius of the dataset (in floor rounded number of FWHM)
    TPF = 0.95 # true positive fraction

    # Then at each radius we have a constant number of false positives
    N_FP_r = np.divide(N_FP_tot,R_max)
    print("N_FP_r:")
    print(N_FP_r)

    # Example pythonic CDF inversion:
    '''
    # generate a t-distribution with 10 DOFs
    df = 10
    mean, var, skew, kurt = t.stats(df, moments='mvsk')

    # print the tau/s-bar where the FPF=0.5 (should be zero!)
    print("PPF where FPF=0.5:")
    print(t.ppf(0.5, df))

    # print the tau/s-bar where the FPF=0.1 (should be negative!)
    # and check it with the cdf
    print("PPF where FPF=0.1:")
    tau_0pt1 = t.ppf(0.1, df)
    print(tau_0pt1)
    print("Check with CDF (should be 0.1):")
    print(t.cdf(tau_0pt1, df))
    '''

    # make a copy of the input dataframe which we will update
    df_corrxn = original_contrast_curve.copy(deep=True)
    # companion amplitude that fits all criteria; i.e., THE CONTRAST CURVE
    df_corrxn["mu_c"] = np.nan
    df_corrxn["mu_c_minus_tau"] = np.nan
    # multiplicative correction factor to 'classical' curve
    df_corrxn["corrxn_factor"] = np.nan
    # max value of FPF (for calculating tau)
    df_corrxn["FPF_tau"] = np.nan
    # min value of TPF, no radial dependency (for calculating tau)
    df_corrxn["TPF"] = np.nan
    # some info in the style of Table 1 in Mawet+ 2014 ApJ 792
    df_corrxn["mu_c_minus_tau"] = np.nan
    df_corrxn["tau"] = np.nan
    df_corrxn["mu_c"] = np.nan
    #df_corrxn["tau_5_sigma"] = np.nan
    #df_corrxn["tau_3_sigma"] = np.nan
    df_corrxn["ppf_1st"] = np.nan
    df_corrxn["ppf_2nd"] = np.nan
    df_corrxn["N_FWHM_fit"] = np.nan
    df_corrxn["F"] = np.nan

    # find the radii in units of decimal lambda/D FWHM
    # (note FWHM=1.028*lambda/D
    #           =(1.028)*(9.463pix)
    #           =0.10409 asec
    #
    df_corrxn["rad_fwhm"] = np.divide(np.divide(df_corrxn["rad_asec"],0.0107),9.728)
    #print(df_corrxn)

    # for each radius, generate a noise parent population and calculate tau
    for rad_num in range(0,len(df_corrxn["rad_fwhm"])):

        # generate a t-distribution at that radius with DOF_r=rad_fwhm_r-2 because the
        # point where the fake planet is located is removed, and 1 is subtracted from
        # those which remain

        # number of whole-number FWHM that can fit in an annulus at that radius
        # (it needs to be the whole-number floor because of how the Altair
        # pipeline samples the noise)
        N_FWHM_fit = math.floor(np.multiply(2*np.pi,df_corrxn["rad_fwhm"][rad_num]))
        # maximum FPF at that radius; i.e., the FPF(r) for the tau we will calculate
        FPF_tau = np.divide(N_FP_r,N_FWHM_fit)
        # sample size of noise 'necklace beads'
        n_2 = N_FWHM_fit-1
        # degrees of freedom of the noise sample
        dof = N_FWHM_fit-2
        print("dof:")
        print(dof)

        # generate a t-distribution for that radius
        mean, var, skew, kurt = t.stats(dof, moments='mvsk')

        # calculate the first and second inverse-CDF terms in the square brackets
        ppf_1st = t.ppf(1-FPF_tau, dof)
        ppf_2nd = t.ppf(TPF, dof)

        # tau
        tau_calc = ppf_1st
        # mu_c-tau
        mu_c_minus_tau = ppf_2nd
        # mu_c alone
        mu_c = np.add(ppf_1st,ppf_2nd)
        # s2: the empirical noise
        # (found by taking companion amplitudes for S/N=5 and dividing by 5)
        s_2 = np.divide(df_corrxn["contrast_lin"].iloc[rad_num],5.)

        # map the mu_c from t-space to amplitude F (F IS THE CONTRAST CURVE)
        F = mu_c*s_2*np.sqrt(1.+(1./n_2))
        # find the correction factor between the original and new contrast curves: alpha = F/A_5
        # N.b. A_5 a.k.a. original_contrast_lin
        corrxn_factor = np.divide(F,df_corrxn["contrast_lin"].iloc[rad_num])

        # now the FPF is smaller due to the offset mu_c-tau; calculate the
        # FPF(t=mu_c) with the final value of tau
        TPF_mu_c_minus_tau = t.cdf(mu_c_minus_tau, dof) # should still be 0.95
        FPF_mu_c = 1.-t.cdf(mu_c, dof)
        print(df_corrxn.keys())

        # update dataframe with new stuff
        df_corrxn.at[rad_num,"mu_c"] = mu_c
        df_corrxn.at[rad_num,"mu_c_minus_tau"] = mu_c_minus_tau
        df_corrxn.at[rad_num,"tau"] = tau_calc
        df_corrxn.at[rad_num,"corrxn_factor"] = corrxn_factor
        df_corrxn.at[rad_num,"FPF_tau"] = FPF_tau
        df_corrxn.at[rad_num,"TPF"] = TPF # should be fixed
        df_corrxn.at[rad_num,"FPF_mu_c"] = FPF_mu_c
        df_corrxn.at[rad_num,"TPF_mu_c_minus_tau"] = TPF_mu_c_minus_tau
        df_corrxn.at[rad_num,"ppf_1st"] = ppf_1st
        df_corrxn.at[rad_num,"ppf_2nd"] = ppf_2nd
        df_corrxn.at[rad_num,"N_FWHM_fit"] = N_FWHM_fit
        df_corrxn.at[rad_num,"F"] = F

    # just to make things clearer
    df_corrxn = df_corrxn.rename(columns={"contrast_lin": "original_contrast_lin"})

    # test with some plots
    # (note the input contrast curve has to be divided by 5 first)
    print("df_corrxn:")
    print(df_corrxn)
    print("df_corrxn.keys():")
    print(df_corrxn.keys())

    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(12,18))

    plt.grid(True, which="both")
    ax[0].plot(df_corrxn["rad_fwhm"], df_corrxn["original_contrast_lin"],'r-', lw=5, alpha=0.6, label="Original contrast curve")
    ax[0].plot(df_corrxn["rad_fwhm"], df_corrxn["F"],'b-', lw=5, alpha=0.6, label="Corrected contrast curve")
    ax[0].set_xlabel("Radius (FWHM)")
    ax[0].set_ylabel("Contrast curves")
    ax[0].set_xlim([0,15])
    ax[0].set_yscale("log")
    ax[0].legend()

    ax[1].plot(df_corrxn["rad_fwhm"], df_corrxn["corrxn_factor"],'g-', lw=5, alpha=0.6, label="Correction factor")
    ax[1].plot(df_corrxn["rad_fwhm"], df_corrxn["mu_c_minus_tau"],'r-', lw=5, alpha=0.6, label="$\mu_{c}-tau$")
    ax[1].set_ylabel("Scaling factors")
    ax[1].set_xlabel("Radius (FWHM)")
    ax[1].set_xlim([0,15])
    ax[1].set_yscale("log")
    ax[1].legend()

    ax[2].plot(df_corrxn["rad_fwhm"], df_corrxn["FPF_tau"],'b-', lw=5, alpha=0.6, label="FPF_tau")
    ax[2].plot(df_corrxn["rad_fwhm"], df_corrxn["FPF_mu_c"],'r-', lw=5, alpha=0.6, label="FPF_mu_c")
    ax[2].set_ylabel("FPF")
    ax[2].set_xlabel("Radius (FWHM)")
    ax[2].set_xlim([0,15])
    ax[2].set_yscale("log")
    ax[2].legend()

    ax[3].plot(df_corrxn["rad_fwhm"], df_corrxn["TPF"],'b-', lw=5, alpha=0.6, label="TPF (input)")
    ax[3].plot(df_corrxn["rad_fwhm"], df_corrxn["TPF_mu_c_minus_tau"],'r-', lw=5, alpha=0.6, label="TPF_mu_c_minus_tau")
    ax[3].set_ylabel("TPF")
    ax[3].set_xlabel("Radius (FWHM)")
    ax[3].set_xlim([0,15])
    ax[3].set_yscale("log")
    ax[3].legend()

    fig.savefig("junk.pdf")


def one_d_classical_contrast(csv_files_dir):
    '''
    Read in the csv with detection information and make a 1D contrast curve

    INPUTS:

    csv_files_dir: directory containing csv files in which it is expected
        to have the keys
        -ampl_linear_norm: linear amplitude of the planet (normalized to star)
        -ampl_linear_norm_0: the starting point linear amplitude (does not change)
        -angle_deg: degrees E of N where planet is injected
        -host_ampl: host star amplitude in counts
        -inject_iteration: iteration of the injection (0 is start)
        -last_ampl_step_unsigned: last change in amplitude taken (from vector in __init__; unsigned)
        -last_ampl_step_signed: same as last_ampl_step_signed, but with sign
        -noise: noise as calculated either from ring or necklace of points
            at planet's radius (but with area around planet itself excluded)
        -signal: max counts in area around injected planet
        -rad_asec: radius from host star of injected planet (asec)
        -rad_pix: same as rad_asec, in pixels
        -s2n: signal/noise
    '''
    # read in config info
    config = configparser.ConfigParser() # for parsing values in .init file
    config.read("./modules/config.ini")
    print(config.items("data_dirs"))

    # read in csv of detection info
    # check for all csvs in the directory
    csv_file_names_array = list(glob.glob(str(csv_files_dir + "/*.csv")))
    print("List of csv files which will be assembled into a contrast curve:")
    print(csv_file_names_array)

    for file_num in range(0,len(csv_file_names_array)):
        # loop over each file and generate the contrast curve data points
        # (these will all be overlapped as the loop finishes)
        info_file = pd.read_csv(csv_file_names_array[file_num])
        if (file_num == 0):
            # initialize a dataframe for containing the most recent (i.e., converged) injections
            df_recent = pd.DataFrame(columns=list(info_file.keys()))
            #import ipdb; ipdb.set_trace()
        # For groups of rows defined by
        #      A.) a common value of rad_asec
        #      B.) a common value of ampl_linear_norm
        # 1. find median value of S/N for each group
        # 2. select lowest value of ampl_linear_norm which provides a minimum X S/N

        # from contrast_curve_1d.ipynb
        # 1. Consider only the rows corresponding to the most recent injection iteration
        #     for each combination of (radius, azimuth, starting amplitude).
        # 2. For each radius, find median value of amplitude across all azimuth and
        #     starting amplitude.

        # find unique combinations of (radius, azimuth, starting amplitude)
        info_file_grouped_rad_ampl_ampl0 = info_file.drop_duplicates(
                                                subset=["rad_asec","angle_deg","ampl_linear_norm_0"]
                                                )
        #import ipdb; ipdb.set_trace()
        for combo_num in range(0,len(info_file_grouped_rad_ampl_ampl0)):
            # loop over each combination of (radius, azimuth, starting amplitude)
            # (i.e., get all the info together corresponding to one fake injected planet)
            info_file_unique_combo = info_file.where(
                np.logical_and(
                    np.logical_and(
                        info_file["rad_asec"] == info_file_grouped_rad_ampl_ampl0["rad_asec"].iloc[combo_num],
                        info_file["angle_deg"] == info_file_grouped_rad_ampl_ampl0["angle_deg"].iloc[combo_num]),
                        info_file["ampl_linear_norm_0"] == info_file_grouped_rad_ampl_ampl0["ampl_linear_norm_0"].iloc[combo_num])
                        ).dropna(how="all")
            #import ipdb; ipdb.set_trace()

            for starting_ampl_num in range(0,len(info_file_grouped_rad_ampl_ampl0["ampl_linear_norm_0"].unique())):
                # loop over all starting amplitudes of this planet (this is redundant
                # unless there are two versions of the same fake planet which are
                # injected with different starting amplitudes)

                # '_0' suffix indicates 'starting amplitude'
                this_ampl = info_file_grouped_rad_ampl_ampl0["ampl_linear_norm_0"].unique()[starting_ampl_num]
                info_file_unique_combo_ampl0 = info_file_unique_combo.where(
                                                    info_file_unique_combo["ampl_linear_norm_0"] == this_ampl
                                                    )
                #import ipdb; ipdb.set_trace()
                # take the row corresponding to the last injection iteration, for THIS starting amplitude
                info_file_unique_combo_ampl0_recent = info_file_unique_combo_ampl0.where(
                                                        info_file_unique_combo_ampl0["inject_iteration"] == np.nanmax(info_file_unique_combo_ampl0["inject_iteration"])
                                                        ).dropna(how="all")

                # add in the file name of the csv file that was output by the pipeline
                info_file_unique_combo_ampl0_recent["original_csv_output_file"] = os.path.basename(csv_file_names_array[file_num])

                # paste this row into the 'new' dataframe
                df_recent = df_recent.append(info_file_unique_combo_ampl0_recent, sort=True)

    print("full table of data, before medianing over radius")
    print(df_recent)

    df_recent = df_recent.reset_index(drop=True)

    # among the rows in the dataframe, take the median at each radius
    # (note that columns with string filenames are lost at this stage)
    # a column of rounded radii is necessary for smooth grouping-by-radius
    df_recent["rad_asec_rounded"] = np.round(df_recent["rad_asec"],4)
    contrast_curve = df_recent.groupby(["rad_asec_rounded"], axis=0, as_index=False).median()

    print("contrast curve; that is, table of data post-median over radii")
    print(contrast_curve)

    # FYI: plot all the points and the final line
    plt.scatter(df_recent["rad_asec"],df_recent["ampl_linear_norm"])
    plt.plot(contrast_curve["rad_asec"],contrast_curve["ampl_linear_norm"])
    plt.savefig("junk.pdf")

    # write out to csv
    #file_name_cc = config["data_dirs"]["DIR_S2N"] + config["file_names"]["CONTCURV_CLASSICAL_CSV"]
    file_name_cc = "contrast_curve_classical.csv"
    contrast_curve.to_csv(file_name_cc, sep = ",", columns = ["rad_asec","ampl_linear_norm"])
    print("sensitivity: "+str(datetime.datetime.now())+" Wrote out contrast curve CSV to " + file_name_cc)

    # make classical contrast curve plot
    #file_name_cc_plot = self.config_data["data_dirs"]["DIR_FYI_INFO"] + self.config_data["file_names"]["CONTCURV_CLASSICAL_PLOT"]
    file_name_cc_plot = "classical_curve.pdf"
    plt.plot(contrast_curve["rad_asec"],contrast_curve["ampl_linear_norm"])
    plt.xlabel("Radius from host star (asec)")
    plt.ylabel("Linear contrast")
    plt.title("Classical contrast curve")
    plt.savefig(file_name_cc_plot)
    plt.clf()
    print("sensitivity: "+str(datetime.datetime.now())+" Wrote out contast curve plot to " + file_name_cc_plot)

    # plot locations of fake planets involved in making the 1D curve
    oversample_factor = 10 # oversample by this much
    # effective plate scale on the display area
    pseudo_ps_LMIR = np.divide(0.0107,oversample_factor)
    # make the array with an odd number of pixels to have a center
    dummy_array_0 = np.nan*np.ones((4001,4001))
    # convert radial info into absolute (y,x)
    y_scatter, x_scatter = convert_rad_xy(
                                dummy_array_0,
                                PS=pseudo_ps_LMIR,
                                rho_asec=df_recent["rad_asec"],
                                theta_deg=df_recent["angle_deg"]
                                )
    ## make a simple scatter plot, where the 3rd dimension is in the marker color
    plt.clf()
    for t_indx in range(0,len(x_scatter)):
        plt.annotate("r="+np.round(df_recent["rad_asec"].iloc[int(t_indx)],2).astype("str")+", rho="+np.round(df_recent["angle_deg"].iloc[int(t_indx)],2).astype("str"),
                    xy=(x_scatter[int(t_indx)],y_scatter[int(t_indx)]),
                    xytext=(x_scatter[int(t_indx)],y_scatter[int(t_indx)]), alpha=0.5
                    )
    plt.scatter(x_scatter, y_scatter, c = df_recent["s2n"])
    plt.title("Signal/Noise")
    plt.ylim([0,np.shape(dummy_array_0)[0]])
    plt.xlim([0,np.shape(dummy_array_0)[1]])
    # compass rose
    plt.annotate("N", xy=(790,410), xytext=(790,410))
    plt.annotate("E", xy=(580,190), xytext=(580,190))
    plt.plot([800,800],[200,400], color="k")
    plt.plot([800,600],[200,200], color="k")
    plt.colorbar()
    # make square
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("junk_2d_fake_planets.pdf")
    #plt.show()
    print("Wrote out scatter signal map")


class TwoDimSensitivityMap:
    '''
    Produces a 2D sensitivity map (for regime of large radii)
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
                 csv_files_dir):
        '''
        Read in the csv with detection information and, for each fake planet amplitude, make a 2D signal map and noise map

        INPUTS:

        csv_file: absolute name of the file which contains the detection information for all fake planet parameters
        '''

        # read in csv of detection info
        info_file = pd.read_csv(csv_file)

        # find unique fake planet amplitudes (on a relative, linear scale)
        unique_ampls = info_file["ampl_linear_norm"].unique()

        # loop over each available fake companion amplitude
        for ampl_num in range(0,len(unique_ampls)):

            # winnow data in the dataframe to involve only this amplitude
            data_right_ampl = info_file.where(info_file["ampl_linear_norm"] == unique_ampls[ampl_num])
            # drop other nans (sometimes signal or noise is nan)
            data_right_ampl = data_right_ampl.dropna()

            print(data_right_ampl)

            ### read in a test science array here to determine size

            oversample_factor = 10 # oversample by this much

            # effective plate scale on the display area
            pseudo_ps_LMIR = np.divide(np.float(self.config_data["instrum_params"]["LMIR_PS"]),oversample_factor)

            # make the array with an odd number of pixels to have a center
            dummy_array_0 = np.nan*np.ones((1001,1001))

            # convert radial info into absolute (y,x)
            y_scatter, x_scatter = convert_rad_xy(dummy_array_0,
                              PS=pseudo_ps_LMIR,
                              rho_asec=data_right_ampl["rad_asec"],
                              theta_deg=data_right_ampl["angle_deg"])

            ## make a simple scatter plot, where the 3rd dimension is in the marker color
            plt.clf()
            plt.scatter(x_scatter, y_scatter, c = data_right_ampl["signal"])
            plt.title("Signal")
            plt.ylim([0,np.shape(dummy_array_0)[0]])
            plt.xlim([0,np.shape(dummy_array_0)[1]])
            # compass rose
            plt.annotate("N", xy=(790,410), xytext=(790,410))
            plt.annotate("E", xy=(580,190), xytext=(580,190))
            plt.plot([800,800],[200,400], color="k")
            plt.plot([800,600],[200,200], color="k")
            # make square
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig("junk_scatter_"+str(unique_ampls[ampl_num])+".pdf")
            print("Wrote out scatter signal map")

            ## make a contour plot where regions between points are interpolated
            # initialize meshgrid
            x_mgrid_range = np.arange(0,np.shape(dummy_array_0)[1])
            y_mgrid_range = np.arange(0,np.shape(dummy_array_0)[0])
            xx, yy = np.meshgrid(x_mgrid_range, y_mgrid_range, sparse=False)
            # interpolate between the empirical points
            print(data_right_ampl["signal"].dropna().values)
            print(data_right_ampl["noise"].dropna().values)
            print(x_scatter.dropna().values)
            print(y_scatter.dropna().values)
            ## ## BEGIN TEST
            '''
            #data_right_ampl["signal"] = np.ones(len(data_right_ampl["signal"]))
            #data_right_ampl["noise"] = 0.1*np.ones(len(data_right_ampl["signal"]))
            x_scatter = 1000*np.random.random(len(x_scatter))
            y_scatter = 1000*np.random.random(len(x_scatter))
            grid_z0_signal = griddata(points=np.transpose([x_scatter,y_scatter]),
                   values=data_right_ampl["signal"].values,
                   xi=(xx, yy),
                   method='cubic')
            grid_z0_noise = griddata(points=np.transpose([x_scatter,y_scatter]),
                   values=data_right_ampl["noise"].values,
                   xi=(xx, yy),
                   method='cubic')
            '''
            ## ## END TEST
            # N.b. In the interpolations, for linear and cubic options to work, x,y sampling
            # has to be heavy enough
            grid_z0_signal = griddata(points=np.transpose([x_scatter,y_scatter]),
                   values=data_right_ampl["signal"].values,
                   xi=(xx, yy),
                   method='linear')
            grid_z0_noise = griddata(points=np.transpose([x_scatter,y_scatter]),
                   values=data_right_ampl["noise"].values,
                   xi=(xx, yy),
                   method='linear')
            plt.clf()
            plt.figure(figsize=(10,5))
            # subplot 1: signal
            plt.subplot(131)
            plt.imshow(grid_z0_signal, origin="lower")
            plt.title("Signal")
            plt.colorbar(fraction=0.046, pad=0.04)
            # compass rose
            plt.annotate("N", xy=(790,410), xytext=(790,410))
            plt.annotate("E", xy=(580,190), xytext=(580,190))
            plt.plot([800,800],[200,400], color="k")
            plt.plot([800,600],[200,200], color="k")
            # overplot discrete points
            plt.scatter(x_scatter, y_scatter, s = 60, c = data_right_ampl["signal"], edgecolors="w")
            # restrict dimensions
            plt.ylim([0,np.shape(dummy_array_0)[0]])
            plt.xlim([0,np.shape(dummy_array_0)[1]])
            plt.gca().set_aspect('equal', adjustable='box')
            # subplot 2: noise
            plt.subplot(132)
            plt.imshow(grid_z0_noise, origin="lower")
            plt.title("Noise")
            plt.colorbar(fraction=0.046, pad=0.04)
            # compass rose
            plt.annotate("N", xy=(790,410), xytext=(790,410))
            plt.annotate("E", xy=(580,190), xytext=(580,190))
            plt.plot([800,800],[200,400], color="k")
            plt.plot([800,600],[200,200], color="k")
            # overplot discrete points
            plt.scatter(x_scatter, y_scatter, s = 60, c = data_right_ampl["noise"], edgecolors="w")
            # restrict dimensions
            plt.ylim([0,np.shape(dummy_array_0)[0]])
            plt.xlim([0,np.shape(dummy_array_0)[1]])
            plt.gca().set_aspect('equal', adjustable='box')
            # subplot 3: signal/noise
            plt.subplot(133)
            plt.imshow(np.divide(grid_z0_signal,grid_z0_noise), origin="lower")
            plt.title("S/N")
            plt.colorbar(fraction=0.046, pad=0.04)
            # compass rose
            plt.annotate("N", xy=(790,410), xytext=(790,410))
            plt.annotate("E", xy=(580,190), xytext=(580,190))
            plt.plot([800,800],[200,400], color="k")
            plt.plot([800,600],[200,200], color="k")
            # overplot discrete points
            plt.scatter(x_scatter, y_scatter, s = 60,
                        c = np.divide(data_right_ampl["signal"],data_right_ampl["noise"]),
                        edgecolors="w")
            # restrict dimensions
            plt.ylim([0,np.shape(dummy_array_0)[0]])
            plt.xlim([0,np.shape(dummy_array_0)[1]])
            plt.gca().set_aspect('equal', adjustable='box')
            s_and_n_map_fyi_file_name = "junk_contour_"+str(unique_ampls[ampl_num])+".pdf"
            plt.tight_layout()
            #ax = plt.gca()
            #divider = make_axes_locatable(ax)
            #cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.savefig(s_and_n_map_fyi_file_name)
            print("Wrote out contour signal and noise map as \n" + s_and_n_map_fyi_file_name)
            print("-"*prog_bar_width)

            # put interpolated data into a cube
            cube_file_name = self.config_data["data_dirs"]["DIR_S2N_CUBES"] + \
              "s_to_n_cube_ampl_" + str(unique_ampls[ampl_num]) + ".fits"
            cube_s_and_n = np.nan*np.ones((2,np.shape(grid_z0_signal)[0],np.shape(grid_z0_signal)[1]))
            cube_s_and_n[0,:,:] = grid_z0_signal
            cube_s_and_n[1,:,:] = grid_z0_noise

            # save interpolated signal and noise data
            print("sensitivity: Writing 3D cube of 2D signal and noise \n" + cube_file_name)
            fits.writeto(filename = cube_file_name,
                     data = cube_s_and_n,
                     header = None,
                     overwrite = True)
            print("-"*prog_bar_width)



def main(small_angle_correction):
    '''
    Detect companions (either fake or in a blind search within science data)
    and calculate S/N.
    '''

    # configuration data
    config = configparser.ConfigParser() # for parsing values in .init file
    config.read("./modules/config.ini")
    print(config.items("data_dirs"))

    # make a 1D contrast curve (implied TPF=0.5) from assemblage of pipeline csv outputs
    # (N.b. this does NOT calculate small angle corrections, FPFs, etc.; it
    # only consolidates the converged S/N=5 fake planet amplitudes)
    one_d_classical_contrast(csv_files_dir = "./csv_outputs/") # placeholder directory

    # now make FPF, TPF, small-angle corrections to make a more modern curve
    #one_d_modern_contrast = OneDimModernContrastCurve()
    '''
    one_d_modern_contrast()

    # make a 2D sensitivity map
    two_d_sensitivity = TwoDimSensitivityMap(csv_file = config["data_dirs"]["DIR_S2N"] + \
        config["file_names"]["DETECTION_CSV"])
    two_d_sensitivity()
    '''
