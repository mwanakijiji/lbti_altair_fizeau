import multiprocessing
import configparser
import glob
import time
from astropy.io import fits
from astropy.convolution import convolve, Gaussian1DKernel, interpolate_replace_nans
from astropy.modeling import models, fitting
from modules import *

# import the PCA machinery for making backgrounds
from .basic_red import BackgroundPCACubeMaker 

import matplotlib
matplotlib.use('agg') # avoids some crashes when multiprocessing
import matplotlib.pyplot as plt


class FakePlanetInjector:
    '''
    PCA-decompose host star PSF and inject fake planet PSFs,
    based on a grid of fake planet parameters

    '''

    def __init__(self,
                 file_list,
                 n_PCA,
                 config_data = config):
        '''
        INPUTS:
        file_list: list of ALL filenames in the directory
        config_data: configuration data, as usual
        '''

        self.file_list = file_list
        self.n_PCA = n_PCA
        self.config_data = config_data

        # read in the PCA vector cube for this series of frames
        
        ##########
        

    def __call__(self,
                 start_frame_num,
                 stop_frame_num):
        '''
        Reconstruct and inject, for a single frame so as to parallelize the job

        INPUTS:
        '''

        # PCA-decompose the host star PSF
        


        
        #

        # inject the fake planet
        # (parameters are:
        # [0]: angle east (in deg) from true north (i.e., after image derotation)
        # [1]: radius (in asec)
        # [2]: contrast ratio (A_star/A_planet, where A_star
        #            is from the PCA_reconstructed star, since the
        #            empirical stellar PSF will be in saturation regime)




        # re-rotate image back to original PA
        
        # add info to the header indicating last reduction step, and PCA info
        #header_sci["RED_STEP"] = "fake_planet_injection"

        # PCA vector file with which the host star was decomposed
        #header_sci["FAKE_PLANET_PCA_CUBE_FILE"] =

        # PCA spectrum of the host star using the above PCA vector, with which the
        # fake planet is injected
        #header_sci["FAKE_PLANET_PCA_SPECTRUM"] = 

        # write file out
        abs_image_cookie_centered_name = str(self.config_data["data_dirs"]["DIR_CENTERED"] + \
                                             os.path.basename(abs_sci_name))
        fits.writeto(filename=abs_image_cookie_centered_name,
                     data=sci_shifted,
                     header=header_sci,
                     overwrite=True)
        print("Writing out centered frame " + os.path.basename(abs_sci_name))
        
        


def main():
    '''
    Carry out steps to write out PSF PCA cubes
    '''

    # configuration data
    config = configparser.ConfigParser() # for parsing values in .init file
    config.read("modules/config.ini")

    # multiprocessing instance
    pool = multiprocessing.Pool(ncpu)
    
    # make a list of the centered cookie cutout files
    cookies_centered_06_directory = str(config["data_dirs"]["DIR_CENTERED"])
    cookies_centered_06_name_array = list(glob.glob(os.path.join(cookies_centered_06_directory, "*.fits")))

    # 
    inject_fake_psfs = FakePlanetInjector()
    pool.map(inject_fake_psfs, cookies_06_name_array) 
