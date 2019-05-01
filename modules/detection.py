import multiprocessing
import configparser
import glob
import time
import pickle
import math
import pandas as pd
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

# -> /blind_search flag: (i.e., we're looking for true candidates)
# 1. do a 2d cross-correlation of the ring with the unsaturated,
#     reconstructed host star PSF (see scipy.signal.correlate2d)
# 2. find where the correlation is maximum
# 3. find the max around that location in the image
# 4. mask that location and find the stdev of the rest of the ring
# 5. if S/N >= Nsigma, flag it!


def circ_mask(input_array, mask_center, mask_radius, invert=False):
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

    mask_array = np.ones(np.shape(input_array))
    y_len = np.arange(np.shape(input_array)[0])
    x_len = np.arange(np.shape(input_array)[1])
    xx, yy = np.meshgrid(x_len, y_len)

    if invert:
        circ_indices = np.where(np.sqrt(np.power(xx-mask_center[1],2) +
                                           np.power(yy-mask_center[0],2)) > mask_radius)
    else:
        circ_indices = np.where(np.sqrt(np.power(xx-mask_center[1],2) +
                                           np.power(yy-mask_center[0],2)) < mask_radius)

    mask_array[circ_indices] = np.nan

    return mask_array


class Median:
    '''
    Derotate frames in series, take median
    '''

    def __init__(self,
                 config_data = config):
        '''
        INPUTS:
        config_data: configuration data, as usual
        '''

        self.config_data = config_data


        ##########


    def __call__(self,abs_sci_name_array):
        '''
        Make the stack and take median

        INPUTS:

        abs_sci_name: the array of absolute paths of the science frames we want to combine
        '''

        print(abs_sci_name_array)

        # read in a first array to get the shape
        shape_test, header = fits.getdata(abs_sci_name_array[0], 0, header=True)

        # initialize a cube
        cube_derotated_frames = np.nan*np.ones((len(abs_sci_name_array),np.shape(shape_test)[0],np.shape(shape_test)[1]))
        del shape_test

        # sort the name array to read in consecutive frames
        sorted_abs_sci_name_array = sorted(abs_sci_name_array)

        # loop over individual frames to derotate them and put them in to a cube
        for t in range(0,len(sorted_abs_sci_name_array)):

            print(t)

            # read in the pre-derotated frames, derotate them, and put them into a cube
            sci, header_sci = fits.getdata(sorted_abs_sci_name_array[t], 0, header=True)

            # derotate
            sci_derotated = scipy.ndimage.rotate(sci, header_sci["LBT_PARA"], reshape=False)

            # put into cube
            cube_derotated_frames[t,:,:] = sci_derotated.astype(np.float32)

        # write cube
        fits.writeto(filename = "junk_stack.fits", data = cube_derotated_frames, overwrite = True)
        print("Wrote cube of derotated frames.")

        # take median and write
        median_stack = np.nanmedian(cube_derotated_frames, axis=0)
        fits.writeto(filename = "junk_median.fits", data = median_stack, overwrite = True)
        print("Wrote median of stack.")


class Detection:
    '''
    Do analysis on the median frame
    '''

    def __init__(self,
                 config_data = config):
        '''
        INPUTS:
        config_data: configuration data, as usual
        '''

        self.config_data = config_data

        # read in the single frame produced by previous module
        ## ## REPLACE FILENAME HERE WITH CONFIG PARAM
        self.master_frame, self.header = fits.getdata("junk_median.fits")

        # radius of aperture around planet candidate (pix)
        self.comp_rad = 10


    def __call__(self):

        # read in a centered PSF model to use for companion search
        ## ## WILL NEED TO CHANGE THIS!
        centered_psf = fits.getdata("lm_180507_009030.fits")

        # if we don't know where a possible companion is, and we're searching blindly for it
        if blind_search:
            
            # find where a companion might be by correlating with centered PSF
            ## ## CHANGE THIS! COMPANION PSF AT LARGE RADII WILL HAVE FRINGES WASHED OUT
            ## ## CORRELATE WITH MAYBE THE MEDIAN OF ALL HOST STARS?
            fake_corr = scipy.signal.correlate2d(self.master_frame, centered_psf, mode="same")

            ## ## BEGIN STAND-IN WITH FAKE PLANET
            fake_planet_standin = np.roll(centered_psf, 30, axis=0)
            fake_planet_standin = np.roll(fake_planet_standin, 25, axis=1)
            fake_planet_frame = np.add(self.master_frame,np.multiply(0.2,fake_planet_standin))
            self.master_frame = fake_planet_frame
            fake_corr = scipy.signal.correlate2d(fake_planet_frame, centered_psf, mode="same")
            ## ## END STAND-IN

            # location of the companion/maximum
            loc_vec = np.where(fake_corr == np.max(fake_corr))

        # if this is a frame involving an injected fake companion, we already know
        # where it is and just want to determine its amplitude relative to the noise
        elif fake_planet:

            # fake planet parameters are in the header
            fake_angle_e_of_n_deg = self.header_sci["FAKEAEON"]
            fake_radius_asec = self.header_sci["FAKERADA"]
            fake_contrast_rel = self.header_sci["FAKECREL"]

            # fake planet injection parameters
            fake_params_pre_permute = {"angle_deg": [0., 60., 120.],\
                               "rad_asec": [0.3, 0.4],\
                               "ampl_linear_norm": [1., 0.9]}

            # location of the fake planet
            injection_loc_dict = {"rad_asec": fake_radius_asec, "angle_deg": fake_angle_e_of_n_deg} # asec, deg E of N
            injection_loc = pd.DataFrame(injection_loc_dict)
            loc_vec = polar_to_xy(pos_info = injection_loc, asec = True)

            ## ## ## THIS IS WHERE I STOPPED; NEED TO INCORPORATE KNOWN AMPLITUDE SOMEHOW? OR STORE IT FOR LATER?
            
        # data type N/A
        else:
            print("Pipeline doesn't know if this data is real or has fake planets!")
            
        # convert to DataFrame
        companion_loc_vec = pd.DataFrame({"y_pix_coord": loc_vec[0], "x_pix_coord": loc_vec[1]})

        # find center of frame for placing of masks
        # N.b. for a 100x100 image, the physical center is at Python coordinate (49.5,49.5)
        # i.e., in between pixels 49 and 50 in both dimensions (Python convention),
        # or at coordinate (50.5,50.5) in DS9 convention
        ## ## check this by displacing, flipping, and subtracting to detect asymmetry
        x_cen = 0.5*np.shape(self.master_frame)[0]-0.5
        y_cen = 0.5*np.shape(self.master_frame)[1]-0.5

        ## ## BEGIN STAND-IN
        pos_num = 0 ## ## stand-in for now; NEED TO CHANGE LATER
        print(loc_vec)
        kernel_scale = 5
        smoothed_w_fake_planet = ndimage.filters.gaussian_filter(self.master_frame,
                                                                 sigma = np.multiply(kernel_scale,[1,1]),
                                                                 order = 0,
                                                                 output = None,
                                                                 mode = "reflect",
                                                                 cval = 0.0,
                                                                 truncate = 4.0)
        ## ## END STAND-IN

        # calculate outer noise annulus radius
        fake_psf_outer_edge_rad = np.add(np.sqrt(np.power(companion_loc_vec["x_pix_coord"][pos_num]-x_cen,2) + 
                                  np.power(companion_loc_vec["y_pix_coord"][pos_num]-y_cen,2)), 
                                  self.comp_rad)

        # calculate inner noise annulus radius
        fake_psf_inner_edge_rad = np.subtract(np.sqrt(np.power(companion_loc_vec["x_pix_coord"][pos_num]-x_cen,2) + 
                                  np.power(companion_loc_vec["y_pix_coord"][pos_num]-y_cen,2)), 
                                  self.comp_rad)

        # invert-mask the companion
        comp_mask_inv = circ_mask(input_array = smoothed_w_fake_planet,
                      mask_center = [companion_loc_vec["y_pix_coord"][pos_num],
                                     companion_loc_vec["x_pix_coord"][pos_num]],
                      mask_radius = self.comp_rad,
                      invert=True)

        # invert-mask the noise ring
        noise_mask_outer_inv = circ_mask(input_array = smoothed_w_fake_planet,
                             mask_center = [y_cen,x_cen],
                             mask_radius = fake_psf_outer_edge_rad,
                             invert=True)
        noise_mask_inner = circ_mask(input_array = smoothed_w_fake_planet,
                             mask_center = [y_cen,x_cen],
                             mask_radius = fake_psf_inner_edge_rad,
                             invert=False)
        comp_mask = circ_mask(input_array = smoothed_w_fake_planet,
                      mask_center = [companion_loc_vec["y_pix_coord"][pos_num],
                                     companion_loc_vec["x_pix_coord"][pos_num]],
                      mask_radius = self.comp_rad,
                      invert=False)

        # mask involving the noise ring without the companion
        net_noise_mask = np.add(np.add(noise_mask_inner,noise_mask_outer_inv),comp_mask)

        # find S/N
        noise_smoothed = np.multiply(smoothed_w_fake_planet,net_noise_mask)
        comp_ampl = np.multiply(smoothed_w_fake_planet,comp_mask_inv)

        print("Signal:")
        print(np.nanmax(comp_ampl))
        print("Noise:")
        print(np.nanstd(noise_smoothed))

        ## BEGIN WRITE OUT AS A CHECK
        fits.writeto(filename="junk_smoothed.fits", data=smoothed_w_fake_planet, overwrite=True)
        fits.writeto(filename="junk_noise_smoothed.fits", data=noise_smoothed, overwrite=True)
        fits.writeto(filename="junk_comp_mask.fits", data=comp_ampl, overwrite=True)
        fits.writeto(filename="junk_host_subt.fits", data=self.master_frame, overwrite=True)
        ## END WRITE OUT AS A CHECK




def main():
    '''
    Detect companions (either fake or in a blind search within science data)
    and calculate S/N.
    '''

    # configuration data
    config = configparser.ConfigParser() # for parsing values in .init file
    config.read("modules/config.ini")

    # multiprocessing instance
    #pool = multiprocessing.Pool(ncpu)

    # make a list of the images WITHOUT fake planets
    hosts_removed_no_fake_psf_08b_directory = str(config["data_dirs"]["DIR_NO_FAKE_PSFS_HOST_REMOVED"])
    hosts_removed_no_fake_psf_08b_name_array = list(glob.glob(os.path.join(hosts_removed_no_fake_psf_08b_directory, "*.fits")))

    # make a median of all frames
    '''
    median_instance = Median()
    make_median = median_instance(hosts_removed_no_fake_psf_08b_name_array)
    '''

    # initialize and parallelize
    detection_blind_search = Detection() #fake=False)

    detection_blind_search()
    #pool.map(detection_blind_search, hosts_removed_no_fake_psf_08b_name_array)
