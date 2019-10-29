import multiprocessing
import configparser
import glob
import time
import pickle
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.interpolate import griddata
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
    array_center = (int(0.5*np.shape(dummy_array_0)[0]),
                    int(0.5*np.shape(dummy_array_0)[1]))
    
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
                 csv_file = config["data_dirs"]["DIR_S2N"] + config["file_names"]["DETECTION_CSV"]):
        '''
        Read in the csv with detection information and make a 1D contrast curve

        INPUTS:

        csv_file: absolute name of the file which contains the detection information for all fake planet parameters
        '''

        # read in csv of detection info
        info_file = pd.read_csv(csv_file)

        # For groups of rows defined by 
        #      A.) a common value of rad_asec
        #      B.) a common value of ampl_linear_norm
        # 1. find median value of S/N for each group
        # 2. select lowest value of ampl_linear_norm which provides a minimum X S/N

        # group by radius and ampl_linear_norm
        info_file_grouped_rad_ampl = info_file.groupby(["rad_asec", "ampl_linear_norm"], 
                                            axis=0, 
                                            as_index=False).median()

        # for each radius, find ampl_linear_norm with S/N > threshold_s2n
        ## ## STAND-IN THRESHOLD S/N
        threshold_s2n = 0.5

        # unique radius values
        unique_rad_vals = info_file_grouped_rad_ampl["rad_asec"].unique()
        print(unique_rad_vals)

        # initialize array for contrast curve
        contrast_curve = {"rad_asec": np.nan*np.ones(len(unique_rad_vals)),
                  "ampl_linear_norm": np.nan*np.ones(len(unique_rad_vals))}
        contrast_curve_pd = pd.DataFrame(contrast_curve)

        # loop over unique radius values
        for t in range(0,len(unique_rad_vals)):
            # subset of data with the right radius from the host star
            data_right_rad = info_file_grouped_rad_ampl.where(\
                                                          info_file_grouped_rad_ampl["rad_asec"] == unique_rad_vals[t]\
                                                          )
            # sub-subset of data with at least the minimum S/N
            data_right_s2n_presort = data_right_rad.where(\
                                                      data_right_rad["s2n"] >= threshold_s2n\
                                                      ).dropna()

            try:
                # the row of data with the minimum S/N above the minimum threshold (if it exists)
                data_right_s2n_postsort = data_right_s2n_presort.where(\
                                                               data_right_s2n_presort["s2n"] == data_right_s2n_presort["s2n"].min()\
                                                               ).dropna()
                '''
                print(data_right_s2n_postsort)
                print(data_right_s2n_postsort["rad_asec"].values[0])
                print(data_right_s2n_postsort["ampl_linear_norm"].values[0])
                '''

                # append companion radius and amplitude values
                print(data_right_s2n_postsort)
                contrast_curve_pd.at[t,"rad_asec"] = data_right_s2n_postsort["rad_asec"].values[0]
                contrast_curve_pd.at[t,"ampl_linear_norm"] = data_right_s2n_postsort["ampl_linear_norm"].values[0]

                '''
                print("------------")
                print(data_right_rad)
                print("-")
                print(data_right_s2n_presort)
                print("-")
                print(data_right_s2n_postsort)
                '''
                
            except:
                print("No data point above min S/N at radius (asec) of " + str(unique_rad_vals[t]))

        # write out to csv
        file_name_cc = config["data_dirs"]["DIR_S2N"] + config["file_names"]["CONTCURV_CSV"]
        contrast_curve_pd.to_csv(file_name_cc, sep = ",", columns = ["rad_asec","ampl_linear_norm"])
        print("Wrote out contrast curve CSV to " + file_name_cc)

        # make plot
        print(contrast_curve_pd)
        file_name_cc_plot = config["data_dirs"]["DIR_FYI_INFO"] + config["file_names"]["CONTCURV_PLOT"]
        plt.plot(contrast_curve_pd["rad_asec"],contrast_curve_pd["ampl_linear_norm"])
        plt.xlabel("Radius from host star (asec)")
        plt.ylabel("Min. companion amplitude with S/N > threshhold")
        plt.savefig(file_name_cc_plot)
        plt.clf()
        print("Wrote out contast curve plot to " + file_name_cc_plot)


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
                 csv_file = config["data_dirs"]["DIR_S2N"] + config["file_names"]["DETECTION_CSV"]):
        '''
        Read in the csv with detection information and, for each fake planet amplitude, make a 2D signal map and noise map

        INPUTS:

        csv_file: absolute name of the file which contains the detection information for all fake planet parameters
        '''

        # read in csv of detection info
        info_file = pd.read_csv(csv_file)

        # find unique fake planet amplitudes (on a relative, linear scale)
        unique_ampls = info_file_grouped_rad_ampl["ampl_linear_norm"].unique()

        # loop over each available amplitude
        for ampl_num in range(0,len(unique_ampls)):

            # winnow data in the dataframe to involve only this amplitude
            data_right_ampl = info_file.where(info_file["ampl_linear_norm"] == unique_ampls[ampl_num])

            print(data_right_ampl)

            ### read in a test science array here to determine size

            oversample_factor = 10 # oversample by this much

            # effective plate scale on the display area
            pseudo_ps_LMIR = np.divide(self.config_data["instrum_params"]["LMIR_PS"],oversample_factor)

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
            plt.ylim([0,np.shape(dummy_array_0)[0]])
            plt.xlim([0,np.shape(dummy_array_0)[1]])
            # compass rose
            plt.annotate("N", xy=(790,410), xytext=(790,410))
            plt.annotate("E", xy=(580,190), xytext=(580,190))
            plt.plot([800,800],[200,400], color="k")
            plt.plot([800,600],[200,200], color="k")
            # make square
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig("junk1.pdf")

            ## make a contour plot where regions between points are interpolated
            # initialize meshgrid
            x_mgrid_range = np.arange(0,np.shape(dummy_array_0)[1])
            y_mgrid_range = np.arange(0,np.shape(dummy_array_0)[0])
            xx, yy = np.meshgrid(x_mgrid_range, y_mgrid_range, sparse=False)
            # interpolate between the empirical points
            grid_z0 = griddata(points=np.transpose([x_scatter,y_scatter]), 
                   values=data_right_ampl["signal"].values, 
                   xi=(xx, yy), 
                   method='nearest')
            plt.clf()
            plt.imshow(grid_z0, origin="lower")
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
            plt.savefig("junk2.pdf")

        


def main():
    '''
    Detect companions (either fake or in a blind search within science data)
    and calculate S/N.
    '''

    # configuration data
    config = configparser.ConfigParser() # for parsing values in .init file
    config.read("modules/config.ini")

    # make a 1D contrast curve
    one_d_contrast = OneDimContrastCurve()
    one_d_contrast()
