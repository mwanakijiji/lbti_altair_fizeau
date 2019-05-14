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
from modules import *




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

            # the row of data with the minimum S/N above the minimum threshold
            data_right_s2n_postsort = data_right_s2n_presort.where(\
                                                               data_right_s2n_presort["s2n"] == data_right_s2n_presort["s2n"].min()\
                                                               ).dropna()
            '''
            print(data_right_s2n_postsort)
            print(data_right_s2n_postsort["rad_asec"].values[0])
            print(data_right_s2n_postsort["ampl_linear_norm"].values[0])
            '''

            # append companion radius and amplitude values
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

        # write out to csv
        file_name_cc = config["data_dirs"]["DIR_S2N"] + config["file_names"]["CONTCURV_CSV"]
        contrast_curve_pd.to_csv(file_name_cc, sep = ",", columns = ["rad_asec","ampl_linear_norm"])
        print("Wrote out contrast curve CSV to " + file_name_cc)

        # make plot
        file_name_cc_plot = config["data_dirs"]["DIR_FYI_INFO"] + config["file_names"]["CONTCURV_PLOT"]
        plt.plot(contrast_curve_pd["rad_asec"],contrast_curve_pd["ampl_linear_norm"])
        plt.xlabel("Radius from host star (asec)")
        plt.ylabel("Min. companion amplitude with S/N > 2")
        plt.savefig(file_name_cc_plot)
        plt.clf()
        print("Wrote out contast curve plot to " + file_name_cc_plot)

        
#class TwoDimSensitivityMap:
#    '''
#    Produces a 2D sensitivity map
#    '''
    
    
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
