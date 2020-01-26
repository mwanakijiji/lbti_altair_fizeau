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
    print('mask center')
    print(mask_center)
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


class MedianCube:
    '''
    Derotate frames in series, take median
    '''

    def __init__(self,
                 fake_params,
                 host_subt_cube,
                 pa_array,
                 frame_array,
                 config_data = config,
                 write_cube = False):
        '''
        INPUTS:
        fake_params: fake planet parameters
        host_subt_cube: cube of host-star-subtracted frames
        pa_array: array of PAs
        config_data: configuration data, as usual
        write_cube: flag as to whether cube of frames should be written to disk (for checking)
        '''

        self.fake_params = fake_params
        self.host_subt_cube = host_subt_cube
        self.pa_array = pa_array
        self.frame_array = frame_array
        self.config_data = config_data
        self.write_cube = write_cube


        ##########


    def __call__(self,
                 apply_mask_after_derot = False,
                 fake_planet = False):
        '''
        Make the stack and take median

        INPUTS:

        write_cube_name: cube of frames to write out if we want to check it
        apply_mask_after_derot: should a mask be generated so as to make bad pixels in a given cube slice NaNs?
        fake_planet: True if there is a fake companion (so we can put the info in the ADI frame header)
        '''

        # initialize cube to contain de-rotated frames
        cube_derotated_frames = np.nan*np.ones(np.shape(self.host_subt_cube))

        # loop over individual slices to derotate them and put them in to a cube
        for t in range(0,np.shape(self.host_subt_cube)[0]):

            # read in the pre-derotated frames, derotate them, and put them into a cube
            sci = self.host_subt_cube[t,:,:]

            # replace nans with zeros to let the rotation work (we'll mask the regions of zeros downstream)
            print("protot")
            sci[~np.isfinite(sci)] = 0

            # derotate according to PA
            sci_derotated = scipy.ndimage.rotate(sci, self.pa_array[t], reshape=False)

            # TEST ONLY
            fits.writeto(filename = "junk_sci.fits",
                         data = sci,
                         overwrite = True)

            ### BEGIN READ IN THE RIGHT MASK
            if apply_mask_after_derot:

                # initialize mask (whether or not it will be needed)
                mask_nan_regions = np.ones(np.shape(sci))

                # choose the mask based on the frame number
                if (self.frame_array[t] <= 7734):
                    nod_position = "nod_up"
                else:
                    nod_position = "nod_down"

                # for Altair 180507 dataset,
                # mask for nod up: mask of borders 10 pixels around the edges
                # mask for nod down: same, but also top-right corner (340:,375:), and bottom 100 pixels
                # convention: 0: bad pixels we will mask; 1: pixels to pass
                if (nod_position == "nod_up"):
                    print("Applying mask to frame after derotation, nod up")
                    mask_nan_regions[:10,:] = 0
                    mask_nan_regions[-10:,:] = 0
                    mask_nan_regions[:,:10] = 0
                    mask_nan_regions[:,-10:] = 0
                elif (nod_position == "nod_down"):
                    print("Applying mask to frame after derotation, nod down")
                    mask_nan_regions[:10,:] = 0
                    mask_nan_regions[-10:,:] = 0
                    mask_nan_regions[:,:10] = 0
                    mask_nan_regions[:,-10:] = 0
                    mask_nan_regions[340:,375:] = 0
                    mask_nan_regions[:100,:] = 0

                # derotate the mask in the same way as the science image
                mask_derotated = scipy.ndimage.rotate(mask_nan_regions, self.pa_array[t], reshape=False)
            
                # multiply the science image by the mask
                # note the derotation causes some of the edge pixels to be neither 0 nor 1
                mask_derotated[np.abs(mask_derotated < 0.5)] = np.nan
                sci_derotated = np.multiply(sci_derotated,mask_derotated)


            # test
            '''
            fits.writeto(filename = "junk_mask_derotated.fits",
                     data = mask_derotated,
                     overwrite = True)
            fits.writeto(filename = "junk_sci_derotated.fits",
                     data = sci_derotated,
                     overwrite = True)
            '''
            ### END READ IN THE RIGHT MASK

            # put into cube
            cube_derotated_frames[t,:,:] = sci_derotated.astype(np.float32)

        # define header
        hdr = fits.Header()
        hdr["ANGEOFN"] = self.fake_params["angle_deg_EofN"]
        hdr["RADASEC"] = self.fake_params["rad_asec"]
        hdr["AMPLIN"] = self.fake_params["ampl_linear_norm"]

        # if writing cube of frames to disk for checking
        #OBSOLETE, SINCE WE ONLY USE THE FINAL ADI FRAMES ANYWAY
        if self.write_cube:

            cube_file_name = self.config_data["data_dirs"]["DIR_OTHER_FITS"] + \
              "cube_just_before_median_ADI_" + \
              str(self.fake_params["angle_deg_EofN"]) + "_" + \
              str(self.fake_params["rad_asec"]) + "_" + \
              str(self.fake_params["ampl_linear_norm"]) + ".fits"
              
            fits.writeto(filename = cube_file_name,
                         data = cube_derotated_frames,
                         header = hdr,
                         overwrite = True)
            print("Wrote cube-just-before-median to disk as " + cube_file_name)

        # take median and write
        median_stack = np.nanmedian(cube_derotated_frames, axis=0)
        adi_file_name = self.config_data["data_dirs"]["DIR_ADI_W_FAKE_PSFS"] + \
          "adi_frame_" + str(self.fake_params["angle_deg_EofN"]) + "_" + \
          str(self.fake_params["rad_asec"]) + "_" + \
          str(self.fake_params["ampl_linear_norm"]) + ".fits"
        fits.writeto(filename = adi_file_name,
                     data = median_stack,
                     header = hdr,
                     overwrite = True)
        print("Wrote median of stack as " + adi_file_name)

        # for memory's sake
        del cube_derotated_frames


class Detection:
    '''
    Do analysis on ONE ADI frame, be it
    1. A frame with a fake planet whose position is known, or
    2. A science frame where we search blindly for a planet
    '''

    def __init__(self,
                 adi_frame_file_name,
                 csv_record_file_name,
                 fake_params = None,
                 config_data = config):
        '''
        INPUTS:
        adi_frame_file_name: absolute name of the ADI frame to be analyzed
        csv_record: absolute name of the csv file in which S/N data is recorded
        fake_params: parameters of a fake planet, if the frame involves a fake planet
        config_data: configuration data, as usual
        '''

        self.fake_params = fake_params
        self.config_data = config_data
        self.adi_frame_file_name = adi_frame_file_name

        # read in the single frame produced by previous module
        ## ## REPLACE FILENAME HERE WITH CONFIG PARAM
        self.master_frame, self.header = fits.getdata(self.adi_frame_file_name, 0, header=True)

        # radius of aperture around planet candidate (pix)
        self.comp_rad = 10

        # csv file to save S/N data
        self.csv_record_file_name = csv_record_file_name


    def __call__(self,
                 blind_search = True):
        '''
        INPUTS:
        blind_search flag: is this a real science frame, where we don't know where a planet is?
        #write: flag as to whether data product should be written to disk (for checking)
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
            print("Location vector of best correlation with PSF template:")
            print(loc_vec)

        # case 2: this is an ADI frame involving an injected fake companion, and we already know
        # where it is and just want to determine its amplitude relative to the noise
        else:

            # fake planet injection parameters in ADI frame are from the header
            # (note units are asec, and deg E of N)
            injection_loc_dict = {"angle_deg": [self.header["ANGEOFN"]],
                                  "rad_asec": [self.header["RADASEC"]],
                                  "ampl_linear_norm": [self.header["AMPLIN"]]}

            print(injection_loc_dict)
            injection_loc = pd.DataFrame(injection_loc_dict)
            injection_loc["angle_deg_EofN"] = injection_loc["angle_deg"] # this step a kludge due to some name changesangle_deg_EofN
            loc_vec = polar_to_xy(pos_info = injection_loc, pa=0, asec = True, south = True) # PA=0 because the frame is derotated
            print("Location vector of fake companion:")
            print(loc_vec)
            
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
        print("comp loc vec")
        print(companion_loc_vec["x_pix_coord"][pos_num])
        print(companion_loc_vec["y_pix_coord"][pos_num])
        print(self.comp_rad)
        #print(np.power(companion_loc_vec["x_pix_coord"][pos_num]-xn,1))
        #print(np.power(companion_loc_vec["y_pix_coord"][pos_num]-y_cen,1))
        fake_psf_outer_edge_rad = np.add(\
                                         np.sqrt(\
                                                 np.power(companion_loc_vec["x_pix_coord"][pos_num],2) + \
                                                 np.power(companion_loc_vec["y_pix_coord"][pos_num],2)\
                                                 ),\
                                                 self.comp_rad)
        print("fake_psf_outer_edge_rad")
        print(fake_psf_outer_edge_rad)

        # calculate inner noise annulus radius
        fake_psf_inner_edge_rad = np.subtract(\
                                         np.sqrt(\
                                                 np.power(companion_loc_vec["x_pix_coord"][pos_num],2) + \
                                                 np.power(companion_loc_vec["y_pix_coord"][pos_num],2)\
                                                 ),\
                                                 self.comp_rad)
        print("fake_psf_inner_edge_rad")
        print(fake_psf_inner_edge_rad)

        # invert-mask the companion
        comp_mask_inv = circ_mask(input_array = smoothed_adi_frame,
                      mask_center = [np.add(y_cen,companion_loc_vec["y_pix_coord"][pos_num]),
                                     np.add(x_cen,companion_loc_vec["x_pix_coord"][pos_num])],
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
                      mask_center = [np.add(y_cen,companion_loc_vec["y_pix_coord"][pos_num]),
                                     np.add(x_cen,companion_loc_vec["x_pix_coord"][pos_num])],
                      mask_radius = self.comp_rad,
                      invert=False)

        # mask involving the noise ring without the companion
        net_noise_mask = np.add(np.add(noise_mask_inner,noise_mask_outer_inv),
                                comp_mask)

        # find S/N
        noise_smoothed = np.multiply(smoothed_adi_frame,net_noise_mask)
        comp_ampl = np.multiply(smoothed_adi_frame,comp_mask_inv)
        signal = np.nanmax(comp_ampl)
        noise = np.nanstd(noise_smoothed)
        s2n = np.divide(signal,noise)

        # append S/N info
        injection_loc_dict["signal"] = signal
        injection_loc_dict["noise"] = noise
        injection_loc_dict["s2n"] = s2n
        
        print("Signal:")
        print(signal)
        print("Noise:")
        print(noise)
        print("S/N:")
        print(s2n)

        # append to csv
        injection_loc_df = pd.DataFrame(injection_loc_dict)
        # check if csv file exists; if it does, don't repeat the header
        exists = os.path.isfile(self.csv_record_file_name)
        injection_loc_df.to_csv(self.csv_record_file_name, sep = ",", mode = "a", header = (not exists))
        print("---------------------")
        print("Appended data to csv ")
        print(str(self.csv_record_file_name))  

        # write out frame as a check
        sn_check_cube = np.zeros((4,np.shape(smoothed_adi_frame)[0],np.shape(smoothed_adi_frame)[1]))
        sn_check_cube[0,:,:] = self.master_frame # the original ADI frame
        sn_check_cube[1,:,:] = smoothed_adi_frame # smoothed frame
        sn_check_cube[2,:,:] = noise_smoothed # the noise ring
        sn_check_cube[3,:,:] = comp_ampl # the area around the companion (be it fake or possibly real)
        fits.writeto(filename = config["data_dirs"]["DIR_S2N_CUBES"] + "sn_check_cube_" + os.path.basename(self.adi_frame_file_name),
                     data = sn_check_cube,
                     overwrite = True)
        print("Wrote out S/N cube for " + os.path.basename(self.adi_frame_file_name))


def main():
    '''
    Detect companions (either fake or in a blind search within science data)
    and calculate S/N.
    '''

    # configuration data
    config = configparser.ConfigParser() # for parsing values in .init file
    config.read("/modules/config.ini")

    ###########################################################
    ## ## IMAGES WITH FAKE PLANETS, TO DETERMINE SENSITIVITY
    
    # make a list of the images WITH fake planets
    hosts_removed_fake_psf_09a_directory = str(config["data_dirs"]["DIR_ADI_W_FAKE_PSFS"])

    # find all combinations of available fake planet parameters using the file names
    hosts_removed_fake_psf_09a_name_array = list(glob.glob(os.path.join(hosts_removed_fake_psf_09a_directory,
                                                                        "*.fits"))) # list of all files
    # list fake planet parameter patterns from adi_frame_xxxxx_xxxxx_xxxxx_lm_YYMMDD_NNNNNN.fits
    print(hosts_removed_fake_psf_09a_name_array[0].split("adi_frame_"))
    degen_param_list = [i.split("adi_frame_")[1].split(".fits")[0] for i in hosts_removed_fake_psf_09a_name_array] # list which may have repeats
    param_list = list(frozenset(degen_param_list)) # remove repeats

    # file which will record all S/N calculations, for each fake planet parameter
    csv_file = config["data_dirs"]["DIR_S2N"] + config["file_names"]["DETECTION_CSV"]

    # check if csv file exists; I want to start with a new one
    exists = os.path.isfile(csv_file)
    if exists:
        input("A fake planet detection CSV file already exists! Hit [Enter] to delete it and continue.")
        os.remove(csv_file)
        
    # loop over all fake planet parameter combinations to retrieve ADI frames and look for signal
    for t in range(0,len(param_list)):

        # extract fake planet parameter raw values as ints
        raw_angle = int(float(param_list[t].split("_")[0]))
        raw_radius = int(float(param_list[t].split("_")[1]))
        raw_contrast = int(float(param_list[t].split("_")[2]))

        # get physical values
        fake_angle_e_of_n_deg = np.divide(raw_angle,100.)
        fake_radius_asec = np.divide(raw_radius,100.)
        fake_contrast_rel = np.power(10.,-np.divide(raw_contrast,100.)) # scale is relative and linear
    
        # specify parameters of fake companion
        fake_params_string = param_list[t]

        # initialize and detect
        detection_blind_search = Detection(adi_frame_file_name = config["data_dirs"]["DIR_ADI_W_FAKE_PSFS"] + \
                                           "adi_frame_"+fake_params_string+".fits",
                                           csv_record_file_name = csv_file)
        detection_blind_search(blind_search = False)
    
    ###########################################################
    ## ## IMAGES WITHOUT FAKE PLANETS; I.E., ACTUAL SCIENCE
        
    # MAKE LIST OF ADI FRAMES IN A DIRECTORY (MAY BE JUST 1)

    # DO CROSS-CORRELATION TO FIND MOST LIKELY SPOT WHERE A PLANET EXISTS

    # FIND THE S/N OF THE 'DETECTION'

    # WRITE DATA TO CSV

    # WHILE S/N >2, DO IT AGAIN (WHILE MASKING THE PRECEDING CANDIDATE FOOTPRINTS)
