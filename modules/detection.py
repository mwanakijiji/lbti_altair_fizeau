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

        # read in a first array to get the shape
        shape_test, header = fits.getdata(abs_sci_name_array[0], 0, header=True)

        # initialize a cube
        cube_derotated_frames = np.nan*np.ones((len(abs_sci_name_array),np.shape(shape_test)[0],np.shape(shape_test)[1]))
        del shape_test

        # sort the name array to read in consecutive frames
        sorted_abs_sci_name_array = sorted(abs_sci_name_array)

        # loop over individual frames to derotate them and put them in to a cube
        for t in range(0,len(sorted_abs_sci_name_array)):

            # read in the pre-derotated frames, derotate them, and put them into a cube
            sci, header_sci = fits.getdata(sorted_abs_sci_name_array[t], 0, header=True)

            # derotate
            sci_derotated = scipy.ndimage.rotate(sci, header_sci["LBT_PARA"], reshape=False)

            # put into cube
            cube_derotated_frames[t,:,:] = sci_derotated.astype(np.float32)

        # generate the header
        hdr_write = fits.Header()
        hdr_write["NUMFRAME"] = np.shape(cube_derotated_frames)[0] # number of frames we're taking median of
        if fake_planet:
            hdr_write["FAKEAEON"] = header_sci["FAKEAEON"] # angle of fake companion, E of N
            hdr_write["FAKERADA"] = header_sci["FAKERADA"] # radial distance of fake companion, asec
            hdr_write["FAKECREL"] = header_sci["FAKECREL"] # contrast ratio of fake companion

        # write cube
        fits.writeto(filename = write_cube_name, data = cube_derotated_frames, header = hdr_write, overwrite = True)
        print("Wrote cube of derotated frames, " + os.path.basename(write_cube_name))

        # take median and write
        median_stack = np.nanmedian(cube_derotated_frames, axis=0)
        fits.writeto(filename = write_adi_name, data = median_stack, header = hdr_write, overwrite = True)
        print("Wrote median of stack, " + os.path.basename(write_adi_name))


class Detection:
    '''
    Do analysis on the median frame
    '''

    def __init__(self,
                 adi_frame_name,
                 config_data = config):
        '''
        INPUTS:
        config_data: configuration data, as usual
        '''

        self.config_data = config_data
        self.adi_frame_name = adi_frame_name

        # read in the single frame produced by previous module
        ## ## REPLACE FILENAME HERE WITH CONFIG PARAM
        self.master_frame, self.header = fits.getdata(self.adi_frame_name, 0, header=True)

        # radius of aperture around planet candidate (pix)
        self.comp_rad = 10


    def __call__(self,
                 blind_search = False,
                 fake_planet = False):
        '''
        INPUTS:
        blind_search/fake_planet flags: these are either/or modes, but both defaults set to false
        '''

        # read in a centered PSF model to use for companion search
        ## ## WILL NEED TO CHANGE THIS!
        centered_psf = fits.getdata("lm_180507_009030.fits")

        # case 1: we don't know where a possible companion is, and we're searching blindly for it
        if blind_search:
            
            # find where a companion might be by correlating with centered PSF
            ## ## CHANGE THIS! COMPANION PSF AT LARGE RADII WILL HAVE FRINGES WASHED OUT
            ## ## CORRELATE WITH MAYBE THE MEDIAN OF ALL HOST STARS?
            fake_corr = scipy.signal.correlate2d(self.master_frame, centered_psf, mode="same")

            # location of the companion/maximum
            loc_vec = np.where(fake_corr == np.max(fake_corr))

        # case 2: this is an ADI frame involving an injected fake companion, and we already know
        # where it is and just want to determine its amplitude relative to the noise
        elif fake_planet:

            # fake planet injection parameters in ADI frame are from the header
            # (note units are asec, and deg E of N)
            injection_loc_dict = {"angle_deg": [self.header["FAKEAEON"]],
                                  "rad_asec": [self.header["FAKERADA"]],
                                  "ampl_linear_norm": [self.header["FAKECREL"]]}

            print(injection_loc_dict)
            injection_loc = pd.DataFrame(injection_loc_dict)
            loc_vec = polar_to_xy(pos_info = injection_loc, asec = True)
            print(loc_vec)
            
        # data type N/A
        else:
            print("Pipeline doesn't know if this data is real or has fake planets!")
            
        # convert to DataFrame
        ## ## note that this is at pixel-level accuracy; refine this later to allow sub-pixel precision
        companion_loc_vec = pd.DataFrame({"y_pix_coord": loc_vec["y_pix_coord"], "x_pix_coord": loc_vec["x_pix_coord"]})

        # find center of frame for placing of masks
        # N.b. for a 100x100 image, the physical center is at Python coordinate (49.5,49.5)
        # i.e., in between pixels 49 and 50 in both dimensions (Python convention),
        # or at coordinate (50.5,50.5) in DS9 convention
        ## ## check this by displacing, flipping, and subtracting to detect asymmetry
        x_cen = 0.5*np.shape(self.master_frame)[0]-0.5
        y_cen = 0.5*np.shape(self.master_frame)[1]-0.5

        ## ## BEGIN STAND-IN
        pos_num = 0 ## ## stand-in for now; NEED TO CHANGE LATER
        kernel_scale = 5
        smoothed_adi_frame = ndimage.filters.gaussian_filter(self.master_frame,
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
        comp_mask_inv = circ_mask(input_array = smoothed_adi_frame,
                      mask_center = [companion_loc_vec["y_pix_coord"][pos_num],
                                     companion_loc_vec["x_pix_coord"][pos_num]],
                      mask_radius = self.comp_rad,
                      invert=True)

        # invert-mask the noise ring
        noise_mask_outer_inv = circ_mask(input_array = smoothed_adi_frame,
                             mask_center = [y_cen,x_cen],
                             mask_radius = fake_psf_outer_edge_rad,
                             invert=True)
        noise_mask_inner = circ_mask(input_array = smoothed_adi_frame,
                             mask_center = [y_cen,x_cen],
                             mask_radius = fake_psf_inner_edge_rad,
                             invert=False)
        comp_mask = circ_mask(input_array = smoothed_adi_frame,
                      mask_center = [companion_loc_vec["y_pix_coord"][pos_num],
                                     companion_loc_vec["x_pix_coord"][pos_num]],
                      mask_radius = self.comp_rad,
                      invert=False)

        # mask involving the noise ring without the companion
        net_noise_mask = np.add(np.add(noise_mask_inner,noise_mask_outer_inv),comp_mask)

        # find S/N
        noise_smoothed = np.multiply(smoothed_adi_frame,net_noise_mask)
        comp_ampl = np.multiply(smoothed_adi_frame,comp_mask_inv)
        signal = np.nanmax(comp_ampl)
        noise = np.nanstd(noise_smoothed)
        s2n = np.divide(signal,noise)
        
        print("Signal:")
        print(signal)
        print("Noise:")
        print(noise)
        print("S/N:")
        print(s2n)

        # write to csv
        ## ## TBD

        # write out as a check
        sn_check_cube = np.zeros((4,np.shape(smoothed_adi_frame)[0],np.shape(smoothed_adi_frame)[1]))
        sn_check_cube[0,:,:] = self.master_frame # the original ADI frame
        sn_check_cube[1,:,:] = smoothed_adi_frame # smoothed frame
        sn_check_cube[2,:,:] = noise_smoothed # the noise ring
        sn_check_cube[3,:,:] = comp_ampl # the area around the companion (be it fake or possibly real)
        fits.writeto(filename = config["data_dirs"]["DIR_S2N_CUBES"] + "sn_check_cube_" + os.path.basename(self.adi_frame_name),
                     data = sn_check_cube,
                     overwrite = True)
        print("Wrote out S/N cube for " + os.path.basename(self.adi_frame_name))


def main():
    '''
    Detect companions (either fake or in a blind search within science data)
    and calculate S/N.
    '''

    # configuration data
    config = configparser.ConfigParser() # for parsing values in .init file
    config.read("modules/config.ini")

    ###########################################################
    ## ## IMAGES WITH FAKE PLANETS, TO DETERMINE SENSITIVITY
    
    # make a list of the images WITH fake planets
    hosts_removed_fake_psf_08a_directory = str(config["data_dirs"]["DIR_FAKE_PSFS_HOST_REMOVED"])

    # find all combinations of available fake planet parameters
    hosts_removed_fake_psf_08a_name_array = list(glob.glob(os.path.join(hosts_removed_fake_psf_08a_directory,"*.fits"))) # list of all files
    # list fake planet parameter patterns from fake_planet_xxxxx_xxxxx_xxxxx_lm_YYMMDD_NNNNNN.fits
    print(hosts_removed_fake_psf_08a_name_array[0].split("fake_planet_"))
    degen_param_list = [i.split("fake_planet_")[1].split("_lm_")[0] for i in hosts_removed_fake_psf_08a_name_array] # list which may have repeats
    param_list = list(frozenset(degen_param_list)) # remove repeats

    # loop over all fake planet parameter combinations and make ADI frames
    '''
    for t in range(0,len(param_list)):

        # extract fake planet parameter raw values as ints
        raw_angle = int(param_list[t].split("_")[0])
        raw_radius = int(param_list[t].split("_")[1])
        raw_contrast = int(param_list[t].split("_")[2])

        # get physical values
        fake_angle_e_of_n_deg = np.divide(raw_angle,100.)
        fake_radius_asec = np.divide(raw_radius,100.)
        fake_contrast_rel = np.power(10.,-np.divide(raw_contrast,100.)) # scale is relative and linear
    
        # specify parameters of fake companion
        fake_params_string = param_list[t]

        # list of files corresponding to that combination of fake planet parameters
        hosts_removed_fake_psf_08a_name_array_one_combo = list(glob.glob(os.path.join(hosts_removed_fake_psf_08a_directory, "*"+fake_params_string+"*.fits")))

        # make a median of all frames
        median_instance = Median()
        make_median = median_instance(abs_sci_name_array = hosts_removed_fake_psf_08a_name_array_one_combo,
                                      write_cube_name = config["data_dirs"]["DIR_ADI_W_FAKE_PSFS_CUBE"] + "cube_"+fake_params_string+".fits",
                                      write_adi_name = config["data_dirs"]["DIR_ADI_W_FAKE_PSFS"] + "median_"+fake_params_string+".fits",
                                      fake_planet = True)
    '''

    # loop again over all fake planet parameter combinations to retrieve ADI frames and look for signal
    for t in range(0,len(param_list)):

        # extract fake planet parameter raw values as ints
        raw_angle = int(param_list[t].split("_")[0])
        raw_radius = int(param_list[t].split("_")[1])
        raw_contrast = int(param_list[t].split("_")[2])

        # get physical values
        fake_angle_e_of_n_deg = np.divide(raw_angle,100.)
        fake_radius_asec = np.divide(raw_radius,100.)
        fake_contrast_rel = np.power(10.,-np.divide(raw_contrast,100.)) # scale is relative and linear
    
        # specify parameters of fake companion
        fake_params_string = param_list[t]

        # initialize and detect
        detection_blind_search = Detection(adi_frame_name = config["data_dirs"]["DIR_ADI_W_FAKE_PSFS"] + "median_"+fake_params_string+".fits")
        detection_blind_search(fake_planet = True)
        #detection_blind_search(blind_search = True)
    
    ###########################################################
    ## ## IMAGES WITHOUT FAKE PLANETS; I.E., ACTUAL SCIENCE
        
    '''
    # make a list of the images WITHOUT fake planets
    hosts_removed_no_fake_psf_08b_directory = str(config["data_dirs"]["DIR_NO_FAKE_PSFS_HOST_REMOVED"])
    hosts_removed_no_fake_psf_08b_name_array = list(glob.glob(os.path.join(hosts_removed_no_fake_psf_08b_directory, "*.fits")))

    # make a median of all frames
    median_instance = Median()
    make_median = median_instance(hosts_removed_no_fake_psf_08b_name_array)
    '''

    # multiprocessing instance
    #pool = multiprocessing.Pool(ncpu)
    #pool.map(detection_blind_search, hosts_removed_no_fake_psf_08b_name_array)
