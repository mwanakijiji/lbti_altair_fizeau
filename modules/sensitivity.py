import multiprocessing
import configparser
import glob
import time
import pickle
import math
import pandas as pd
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
                 abs_sci_name_array,
                 write_cube_name,
                 write_adi_name,
                 fake_planet = False):
        '''
        Make the stack and take median

        INPUTS:

        abs_sci_name: the array of absolute paths of the science frames we want to combine
        write_adi_name: absolute path filename for writing out the ADI frame
        fake_planet: True if there is a fake companion (so we can put the info in the ADI frame header)
        '''

        
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

    # read in csv of detection info
    csv_detection = config["data_dirs"]["DIR_S2N"] + config["file_names"]["DETECTION_CSV"]
