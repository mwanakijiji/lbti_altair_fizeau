import multiprocessing
import configparser
import glob
import time
import pickle
import math
from astropy.io import fits
from modules import *

# import the PCA machinery for making backgrounds
from .basic_red import BackgroundPCACubeMaker

import matplotlib
matplotlib.use('agg') # avoids some crashes when multiprocessing
import matplotlib.pyplot as plt

# First part reads in a stack of images from which
# 1. host star has been subtracted
# 2. images have been de-rotated
# 3. a fake planet may or may not be present

# So the task here is to
# 1. median the stack
# 2. convolve the median to smooth it

# ... and, if there is a

# -> /fake_planet flag: (i.e., we're determining sensitivity)
# 1. given the fake planet location, find its amplitude
# 2. find the stdev of the noise ring
# 3. count number of other false positives of amplitude >=Nsigma
# 4. calculate false positive fraction (FPF)

# -> /true_data flag: (i.e., we're looking for true candidates)
# 1. do a 2d cross-correlation of the ring with the unsaturated,
#     reconstructed host star PSF (see scipy.signal.correlate2d)
# 2. find where the correlation is maximum
# 3. find the max around that location in the image
# 4. mask that location and find the stdev of the rest of the ring
# 5. if S/N >= Nsigma, flag it!


class HostRemoval:
    '''
    PCA-decompose a saturated host star PSF and remove it
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
                 abs_sci_name_array):
        '''
        Detect companions, for a single frame so as to parallelize the job

        INPUTS:

        abs_sci_name: the array of absolute paths of the science frames we want to combine
        '''

        print(abs_sci_name_array)

        # read in a first array to get the shape
        shape_test, header = fits.getdata(abs_sci_name_array[0], 0, header=True)

        # initialize a cube
        cube_derotated_frames = np.zeros((len(abs_sci_name_array),np.shape(shape_test)[0],np.shape(shape_test)[1]))
        del shape_test

        # sort the name array to read in consecutive frames
        sorted_abs_sci_name_array = sorted(abs_sci_name_array)

        for t in range(0,len(sorted_abs_sci_name_array)):

            # read in the pre-derotated frames, derotate them, and put them into a cube
            sci, header_sci = fits.getdata(sorted_abs_sci_name_array[t], 0, header=True)

            # derotate
            sci_derotated = scipy.ndimage.rotate(sci, header_sci["LBT_PARA"], reshape=False)

            # put into cube
            cube_derotated_frames[t,:,:] = sci_derotated


        print("Made cube of derotated frames.")




def main():
    '''
    Detect companions (either fake or in a blind search within science data)
    and calculate S/N.
    '''

    # configuration data
    config = configparser.ConfigParser() # for parsing values in .init file
    config.read("modules/config.ini")

    # multiprocessing instance
    pool = multiprocessing.Pool(ncpu)

    # make a list of the images WITHOUT fake planets
    hosts_removed_no_fake_psf_08b_directory = str(config["data_dirs"]["DIR_NO_FAKE_PSFS_HOST_REMOVED"])
    hosts_removed_no_fake_psf_08b_name_array = list(glob.glob(os.path.join(hosts_removed_no_fake_psf_08b_directory, "*.fits")))

    # initialize and parallelize
    detection_blind_search = Detection(fake=False)

    pool.map(detection_blind_search, hosts_removed_no_fake_psf_08b_name_array)
