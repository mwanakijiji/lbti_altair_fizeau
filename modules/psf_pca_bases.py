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


class PSFPCACubeMaker:
    '''
    Make PCA bases of PSFs, in style of background PCA decomposition
    but with different masks (or no masks), no subtraction of background
    (i.e., it is assumed to be zero), and there is no consideration of
    channel variations

    INHERITANCE:
    BackgroundPCACubeMaker
    '''

    def __init__(self,
                 file_list,
                 n_PCA,
                 config_data = config):
        '''
        INPUTS:
        file_list: list of ALL filenames in the directory
        n_PCA: number of PCA components to save in the cube
        -> (does NOT include possible separate components representing
        -> individual channel variations)
        config_data: configuration data, as usual
        '''

        self.file_list = file_list
        self.n_PCA = n_PCA
        self.config_data = config_data
        

    def __call__(self,
                 start_frame_num,
                 stop_frame_num):
        '''
        Make PCA cube (for future step of reconstructing) a PSF

        INPUTS:
        start_frame_num: starting frame number to use in PCA basis generation
        stop_frame_num: stopping [ditto], inclusive
        '''
        
        # read in a first file to get the shape
        test_img, header = fits.getdata(self.file_list[0], 0, header=True)
        shape_img = np.shape(test_img)
        
        print("Initializing a PCA cube...")
        training_cube = np.nan*np.ones((stop_frame_num-start_frame_num+1,shape_img[0],shape_img[1]), dtype = np.int64)

        ## ## MASKING DEACTIVATED FOR THE MOMENT; IN FACT IT MIGHT NOT BE NEEDED AT THIS STEP
        #mask_weird = make_first_pass_mask(quad_choice) # make the right mask
        mask_weird = np.ones(shape_img)

        # loop over frames to add them to training cube
        for frame_num in range(start_frame_num, stop_frame_num+1):

            # get name of file that this number corresponds to
            abs_matching_file_array = [s for s in self.file_list if str("{:0>6d}".format(frame_num)) in s]
            abs_matching_file = abs_matching_file_array[0] # get the name
            
            # if there was a match
            if (len(abs_matching_file) != 0):

                # read in the science frame from raw data directory
                sci, header_sci = fits.getdata(abs_matching_file, 0, header=True)

                # add to cube
                training_cube[frame_num-start_frame_num,:,:] = sci

            # if there was no match
            elif (len(abs_matching_file) == 0):

                print("Frame " + os.path.basename(abs_matching_file) + " not found.")

            # if there were multiple matches
            else:

                print("Something is amiss with your frame number choice.")
                break

        # mask the raw training set
        training_cube_masked_weird = np.multiply(training_cube,mask_weird)
        del training_cube

        # generate the PCA cube from the PSF data
        pca_comp_cube = PCA_basis(training_cube_masked_weird, n_PCA = self.n_PCA)
        
        # write out the PCA vector cube
        abs_pca_cube_name = str(self.config_data["data_dirs"]["DIR_OTHER_FITS"] +
                                'psf_PCA_vector_cookie_' +
                                '_seqStart_'+str("{:0>6d}".format(start_frame_num))+
                                '_seqStop_'+str("{:0>6d}".format(stop_frame_num))+'.fits')
        fits.writeto(filename=abs_pca_cube_name,
                     data=pca_comp_cube,
                     header=None,
                     overwrite=True)
        print("Wrote out PSF PCA cube " + os.path.basename(abs_pca_cube_name))


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

    # generate PCA cubes for PSFs
    # (N.b. n_PCA needs to be smaller that the number of frames being used)
    ## WILL NEED MORE PARAMETER ARRAYS FOR VARIOUS FRAME SEQUENCES
    pca_psf_maker = PSFPCACubeMaker(file_list = cookies_centered_06_name_array,
                                    n_PCA = 10) # create instance
    pca_psf_maker(start_frame_num = 9000,
                   stop_frame_num = 9099)
