import multiprocessing
import configparser
import glob
import time
import pickle
import math
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
            - it is expected this file will have keys
            ampl_linear_norm: linear amplitude of the planet (normalized to star)
            ampl_linear_norm_0: the starting point linear amplitude (does not change)
            angle_deg: degrees E of N where planet is injected
            host_ampl: host star amplitude in counts
            inject_iteration: iteration of the injection (0 is start)
            last_ampl_step_unsigned: last change in amplitude taken (from vector in __init__; unsigned)
            last_ampl_step_signed: same as last_ampl_step_signed, but with sign
            noise: noise as calculated either from ring or necklace of points
                at planet's radius (but with area around planet itself excluded)
            signal: max counts in area around injected planet
            rad_asec: radius from host star of injected planet (asec)
            rad_pix: same as rad_asec, in pixels
            s2n: signal/noise
        '''

        # read in csv of detection info
        info_file = pd.read_csv(csv_file)

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

        ### BEGIN PASTED
        # find unique combinations of (radius, azimuth, starting amplitude)
        info_file_grouped_rad_ampl_ampl0 = info_file.drop_duplicates(subset=["rad_asec",
                                                                     "angle_deg",
                                                                     "ampl_linear_norm_0"])
        # initialize a dataframe for containing the most recent injections
        df_recent = pd.DataFrame(columns=list(info_file_grouped_rad_ampl_ampl0.keys()))

        for combo_num in range(0,len(info_file_grouped_rad_ampl_ampl0)):
            # loop over each combination of (radius, azimuth, starting amplitude)
            info_file_unique_combo = info_file.where(
                np.logical_and(
                    np.logical_and(
                        info_file["rad_asec"] == info_file_grouped_rad_ampl_ampl0["rad_asec"].iloc[combo_num],
                        info_file["angle_deg"] == info_file_grouped_rad_ampl_ampl0["angle_deg"].iloc[combo_num]),
                        info_file["ampl_linear_norm_0"] == info_file_grouped_rad_ampl_ampl0["ampl_linear_norm_0"].iloc[combo_num])
                        ).dropna(how="all")

            for starting_ampl_num in range(0,len(info_file_grouped_rad_ampl_ampl0["ampl_linear_norm_0"].unique())):
                # loop over all starting amplitudes
                this_ampl = info_file_grouped_rad_ampl_ampl0["ampl_linear_norm_0"].unique()[starting_ampl_num]
                info_file_unique_combo_ampl0 = info_file_unique_combo.where(info_file_unique_combo["ampl_linear_norm_0"] ==
                                                                    this_ampl)

                # take the row corresponding to the last injection iteration, for THIS starting amplitude
                info_file_unique_combo_ampl0_recent = info_file_unique_combo_ampl0.where(info_file_unique_combo_ampl0["inject_iteration"] ==
                                                                                 np.nanmax(info_file_unique_combo_ampl0["inject_iteration"])).dropna(how="all")

                # paste this row into the 'new' dataframe
                df_recent = df_recent.append(info_file_unique_combo_ampl0_recent, sort=True)

        ### END PASTED
        # write out to csv
        file_name_cc = config["data_dirs"]["DIR_S2N"] + config["file_names"]["CONTCURV_CSV"]
        contrast_curve_pd.to_csv(file_name_cc, sep = ",", columns = ["rad_asec","ampl_linear_norm"])
        print("sensitivity: "+str(datetime.datetime.now())+" Wrote out contrast curve CSV to " + file_name_cc)

        # make plot
        print(contrast_curve_pd)
        file_name_cc_plot = config["data_dirs"]["DIR_FYI_INFO"] + config["file_names"]["CONTCURV_PLOT"]
        plt.plot(contrast_curve_pd["rad_asec"],contrast_curve_pd["ampl_linear_norm"])
        plt.xlabel("Radius from host star (asec)")
        plt.ylabel("Min. companion amplitude with S/N > threshhold")
        plt.savefig(file_name_cc_plot)
        plt.clf()
        print("sensitivity: "+str(datetime.datetime.now())+" Wrote out contast curve plot to " + file_name_cc_plot)


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
                 csv_file):
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
    config.read("/modules/config.ini")

    # make a 1D contrast curve
    one_d_contrast = OneDimContrastCurve(csv_file = config["data_dirs"]["DIR_S2N"] + \
        config["file_names"]["DETECTION_CSV"])
    one_d_contrast()

    '''
    # make a 2D sensitivity map
    two_d_sensitivity = TwoDimSensitivityMap(csv_file = config["data_dirs"]["DIR_S2N"] + \
        config["file_names"]["DETECTION_CSV"])
    two_d_sensitivity()
    '''
