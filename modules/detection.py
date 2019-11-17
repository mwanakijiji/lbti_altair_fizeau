import multiprocessing
import configparser
import glob
import time
import pickle
import math
import sys
import datetime
import pandas as pd
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from scipy.signal import convolve as scipy_convolve
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
                injection_iteration,
                 fake_params,
                 host_subt_cube,
                 pa_array,
                 frame_array,
                 config_data = config,
                 write_cube = False):
        '''
        INPUTS:
        injection_iteration: iteration of fake planet injections (None: there is no fake planet)
        fake_params: fake planet parameters
        host_subt_cube: cube of frames to derotate and median (in context, this may or may not
            mean the host-star-subtracted frames
        pa_array: array of PAs
        frame_array: array of frame numbers
        config_data: configuration data, as usual
        write_cube: flag as to whether cube of frames should be written to disk (for checking)
        '''

        self.injection_iteration = injection_iteration
        self.fake_params = fake_params
        self.host_subt_cube = host_subt_cube
        self.pa_array = pa_array
        self.frame_array = frame_array
        self.config_data = config_data
        self.write_cube = write_cube


        ##########


    def __call__(self,
                 adi_write_name = None,
                 apply_mask_after_derot = False,
                 fake_planet = False):
        '''
        Make the stack and take median

        INPUTS:

        adi_write_name: file name of written ADI frame; if None, a default will be used
        apply_mask_after_derot: should a mask be generated so as to make bad pixels in a given cube slice NaNs?
        fake_planet: True if there is a fake companion (so we can put the info in the ADI frame header)
        '''

        # string for making subdirectories to place ADI frames in
        if (self.injection_iteration is not None):
            injection_iteration_string = "inj_iter_" + str(self.injection_iteration).zfill(4)
        else:
            injection_iteration_string = "no_fake_planet"

        # initialize cube to contain de-rotated frames
        cube_derotated_frames = np.nan*np.ones(np.shape(self.host_subt_cube))

        # loop over individual slices to derotate them and put them in to a cube
        for t in range(0,np.shape(self.host_subt_cube)[0]):

            # read in the pre-derotated frames, derotate them, and put them into a cube
            sci = self.host_subt_cube[t,:,:]

            # replace nans with zeros to let the rotation work (we'll mask the regions of zeros downstream)
            sci[~np.isfinite(sci)] = 0

            # derotate according to PA
            sci_derotated = scipy.ndimage.rotate(sci, self.pa_array[t], reshape=False)

            # TEST ONLY
            '''
            fits.writeto(filename = "junk_sci.fits",
                         data = sci,
                         overwrite = True)
            '''

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
                    print("detection: Applying mask to frame after derotation, nod up")
                    mask_nan_regions[:10,:] = 0
                    mask_nan_regions[-10:,:] = 0
                    mask_nan_regions[:,:10] = 0
                    mask_nan_regions[:,-10:] = 0
                elif (nod_position == "nod_down"):
                    print("detection: Applying mask to frame after derotation, nod down")
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
            print("detection: "+str(datetime.datetime.now())+": Wrote cube-just-before-median to disk as " + cube_file_name)

        # take median and write
        median_stack = np.nanmedian(cube_derotated_frames, axis=0)
        # check if injection_iteration_string directory exists; if not, make the directory
        abs_path_name = self.config_data["data_dirs"]["DIR_ADI_W_FAKE_PSFS"] + \
            injection_iteration_string + "/"
        if not os.path.exists(abs_path_name):
            os.makedirs(abs_path_name)
            print("Made directory " + abs_path_name)
        if (adi_write_name == None):
            # default name, corresponding to an ADI frame on which to do science
            adi_file_name = self.config_data["data_dirs"]["DIR_ADI_W_FAKE_PSFS"] + \
                injection_iteration_string + "/" + \
              "adi_frame_" + str(self.fake_params["angle_deg_EofN"]) + "_" + \
              str(self.fake_params["rad_asec"]) + "_" + \
              str(self.fake_params["ampl_linear_norm"]) + ".fits"
        else:
            # if it is some other median we want to save, use the user-given name
            adi_file_name = adi_write_name
        import ipdb; ipdb.set_trace()

        fits.writeto(filename = adi_file_name,
                     data = median_stack,
                     header = hdr,
                     overwrite = True)
        print("detection: "+str(datetime.datetime.now())+": Wrote median of stack as " + adi_file_name)
        print("-"*prog_bar_width)

        # for memory's sake
        del cube_derotated_frames


class Detection:
    '''
    Do analysis on ONE ADI frame, be it
    1. A frame with a fake planet whose position is known, or
    2. A science frame where we search blindly for a planet
    '''

    def __init__(self,
                injection_iteration,
                 adi_frame_file_name,
                 csv_record_file_name,
                 fake_params = None,
                 config_data = config,
                 inject_iteration = None):
        '''
        INPUTS:
        injection_iteration: number of fake planet injection iteration (None: no fake planet)
        adi_frame_file_name: absolute name of the ADI frame to be analyzed
        csv_record: absolute name of the csv file in which S/N data is recorded
        fake_params: parameters of a fake planet, if the frame involves a fake planet
        config_data: configuration data, as usual
        inject_iteration: iteration for injecting fake planets
        '''

        self.injection_iteration = injection_iteration
        self.fake_params = fake_params
        self.config_data = config_data
        self.adi_frame_file_name = adi_frame_file_name
        self.inject_iteration = inject_iteration

        # read in the single frame produced by previous module
        self.master_frame, self.header = fits.getdata(self.adi_frame_file_name, 0, header=True)

        # radius of aperture around planet candidate (pix)
        self.comp_rad = 0.5*fwhm_4um_lbt_airy_pix

        # csv file to save S/N data
        self.csv_record_file_name = csv_record_file_name


    def __call__(self,
                 sci_median_file_name,
                 noise_option = "full_ring",
                 noise_annulus_half_width_pix = 0.5*fwhm_4um_lbt_airy_pix,
                 blind_search = True):
        '''
        INPUTS:
        sci_median_file_name: name of file which will be used to find host star amplitude
        noise_option:
            "full_ring"- calculate the noise using the rms of the whole smoothed annulus (minus companion location)
            "necklace"- calculate the noise using the rms of the medians of patches within ring where
                        companions could be
        noise_annulus_half_width_pix: 0.5*thickness of noise annulus ring (if noise_option=="full_ring")
        blind_search flag: is this a real science frame, where we don't know where a planet is?
        #write: flag as to whether data product should be written to disk (for checking)
        '''

        # read in a centered PSF model to use for companion search
        ## ## WILL NEED TO CHANGE THIS!
        print("Reading in centered PSF model to use for companion search:")
        centered_psf_model_file_name = "lm_180507_009030.fits"
        print(centered_psf_model_file_name)
        print("-"*prog_bar_width)
        centered_psf = fits.getdata(centered_psf_model_file_name)

        # case 1: we don't know where a possible companion is, and we're searching blindly for it
        if blind_search:

            injection_iteration_string = "no_fake_planet"

            # find where a companion might be by correlating with centered PSF
            ## ## CHANGE THIS! COMPANION PSF AT LARGE RADII WILL HAVE FRINGES WASHED OUT
            ## ## CORRELATE WITH MAYBE THE MEDIAN OF ALL HOST STARS?
            fake_corr = scipy.signal.correlate2d(self.master_frame, centered_psf, mode="same")

            # location of the companion/maximum
            loc_vec = np.where(fake_corr == np.max(fake_corr))
            print("Location vector of best correlation with PSF template:")
            print(loc_vec)

            # THIS WILL NEED TO BE FOLLOWED WITH A NEXT ITERATION FOR WHERE A COMPANION MAY LIE

        # case 2: this is an ADI frame involving an injected fake companion, and we already know
        # where it is and just want to determine its amplitude relative to the noise
        else:

            injection_iteration_string = "inj_iter_" + str(self.injection_iteration).zfill(4)

            # fake planet injection parameters in ADI frame are from the header
            # (note units are asec, and deg E of N)
            injection_loc_dict = {"angle_deg": [self.header["ANGEOFN"]],
                                  "rad_asec": [self.header["RADASEC"]],
                                  "ampl_linear_norm": [self.header["AMPLIN"]]}

            print(injection_loc_dict)
            injection_loc = pd.DataFrame(injection_loc_dict)
            injection_loc["angle_deg_EofN"] = injection_loc["angle_deg"] # this step a kludge due to some name changes
            loc_vec = polar_to_xy(pos_info = injection_loc, pa=0, asec = True, south = True) # PA=0 because the frame is derotated
            print("Location vector of fake companion:")
            print(loc_vec)

        # convert to DataFrame
        ## ## note that this is at pixel-level accuracy; refine this later to allow sub-pixel precision
        companion_loc_vec = pd.DataFrame({"y_pix_coord": loc_vec["y_pix_coord"],
                                          "x_pix_coord": loc_vec["x_pix_coord"]})

        # find center of frame for placing of masks
        # N.b. for a 100x100 image, the physical center is at Python coordinate (49.5,49.5)
        # i.e., in between pixels 49 and 50 in both dimensions (Python convention),
        # or at coordinate (50.5,50.5) in DS9 convention
        ## ## check this by displacing, flipping, and subtracting to detect asymmetry
        x_cen = 0.5*np.shape(self.master_frame)[0]-0.5
        y_cen = 0.5*np.shape(self.master_frame)[1]-0.5

        # read in median science frame for determining host star amplitude
        print("Reading in median science frame for determining host star amplitude from")
        print(sci_median_file_name)
        print("-"*prog_bar_width)
        sci_median_frame = fits.getdata(sci_median_file_name, 0, header=False)
        pos_num = 0 ## ## stand-in for now; NEED TO CHANGE LATER
        kernel = Gaussian2DKernel(x_stddev=0.5*fwhm_4um_lbt_airy_pix)
        print("Convolving ADI and median science frames with same kernel")
        print("-"*prog_bar_width)
        smoothed_sci_median_frame = convolve(sci_median_frame, kernel) # smooth sci frame with same kernel
        smoothed_adi_frame = convolve(self.master_frame, kernel)
        #smoothed_adi_frame = self.master_frame ## ## NO SMOOTHING AT ALL

        # find amplitude of host star in SMOOTHED image
        center_sci_median_frame = [int(0.5*np.shape(sci_median_frame)[0]),
                                   int(0.5*np.shape(sci_median_frame)[1])]
        host_ampl = np.nanmax(smoothed_sci_median_frame[center_sci_median_frame[0]-10:center_sci_median_frame[0]+10,
                                               center_sci_median_frame[1]-10:center_sci_median_frame[1]+10])
        print("host_ampl")
        print(host_ampl)
        # calculate outer noise annulus radius
        print("comp loc vec from center")
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
                                                 noise_annulus_half_width_pix)
        print("fake_psf_outer_edge_rad")
        print(fake_psf_outer_edge_rad)

        # calculate inner noise annulus radius
        fake_psf_inner_edge_rad = np.subtract(\
                                         np.sqrt(\
                                                 np.power(companion_loc_vec["x_pix_coord"][pos_num],2) + \
                                                 np.power(companion_loc_vec["y_pix_coord"][pos_num],2)\
                                                 ),\
                                                 noise_annulus_half_width_pix)
        print("fake_psf_inner_edge_rad")
        print(fake_psf_inner_edge_rad)

        # invert-mask the companion
        # (i.e., 1s over the companion, NaNs everywhere else)
        comp_mask_inv = circ_mask(input_array = smoothed_adi_frame,
                      mask_center = [np.add(y_cen,companion_loc_vec["y_pix_coord"][pos_num]),
                                     np.add(x_cen,companion_loc_vec["x_pix_coord"][pos_num])],
                      mask_radius = self.comp_rad,
                      invert=True)

        ## one method of finding noise: get the medians of companion-sized patches in a necklace pattern
        # first, calculate the positions where other non-overlapping companions could fit within noise annulus
        # find circumference of circle (in pix) at the radius of the companion
        r_pix = np.divide(injection_loc_dict["rad_asec"][0],np.float(self.config_data["instrum_params"]["LMIR_PS"]))
        circ_length = 2*np.pi*r_pix
        # whole number of companions that fit inside that circle, minus the companion itself
        num_other_comps = np.floor(np.divide(circ_length,2*self.comp_rad) - 1)
        # find angles of the other patches, going deg E of the companion
        # angular offset between patches: theta_step = l_step/r
        angle_offset = np.divide(2*self.comp_rad,r_pix)*np.divide(360.,2*np.pi)
        other_angles = np.mod(injection_loc_dict["angle_deg"] + angle_offset*np.arange(1,num_other_comps+1), 360)
        # initialize array to hold values from each patch
        patch_center_array = np.nan*np.ones(len(other_angles))
        patch_num = 0 # kludge for making sure we start at zero
        for patch_num in range(0,len(other_angles)):
            # convert patch positions to x,y coordinates
            patch_loc_dict = {"angle_deg_EofN": other_angles[patch_num],
                                  "rad_asec": [self.header["RADASEC"]],
                                  "ampl_linear_norm": [self.header["AMPLIN"]]}
            patch_loc_vec = polar_to_xy(pos_info = patch_loc_dict, pa=0, asec = True, south = True) # PA=0 because the frame is derotated
            # make mask for each necklace patch
            necklace_patch_mask_inv = circ_mask(input_array = smoothed_adi_frame,
                              mask_center = [np.add(y_cen,patch_loc_vec["y_pix_coord"][0]),
                                             np.add(x_cen,patch_loc_vec["x_pix_coord"][0])],
                                             mask_radius = 0.75,
                                             invert=True)
            # determine the median value in the patch
            noise_smoothed_patch = np.multiply(smoothed_adi_frame,necklace_patch_mask_inv)
            center_of_patch = np.nanmedian(noise_smoothed_patch)

            # put into array
            patch_center_array[patch_num] = center_of_patch

            # construct array to show necklace of patches
            if patch_num == 0:
                # initialize
                necklace_2d_array = np.zeros( np.shape(noise_smoothed_patch) )
            necklace_2d_array = np.nansum( np.dstack((necklace_2d_array,noise_smoothed_patch)),2 )

        # BEGIN TEST
        if (len(other_angles) > 1): # at small radii, there is not enough room for a necklace of patches
            print("patch num")
            print(patch_num)
            plt.imshow(necklace_2d_array, origin="lower")
            plt.colorbar()
            plt.savefig("junk_necklace.png")
        # END TEST

        # at this point, the array of median values of necklaced patches has been made

        ## another method of finding noise: take stdev of a smoothed ring at same radius of companion
        # first, invert-mask the noise ring
        # (i.e., 1s in the ring, NaNs everywhere else)
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

        # BEGIN TEST
        plt.clf()
        plt.imshow(comp_mask, origin="lower")
        plt.colorbar()
        plt.savefig("junk_comp_mask.png")
        # END TEST

        # mask involving the noise ring without the companion
        net_noise_mask = np.add(np.add(noise_mask_inner,noise_mask_outer_inv),
                                comp_mask)
        # this addition of masks leads to values >1, so
        # 1. do sanity check that the finite values are the same
        # 2. normalize
        if ((np.nanmax(net_noise_mask) == np.nanmin(net_noise_mask)) and
            np.isfinite(np.nanmax(net_noise_mask))):
            # if finite values are the same
            net_noise_mask = np.divide(net_noise_mask,np.nanmax(net_noise_mask))
        elif (np.isfinite(np.nanmax(net_noise_mask) == False) and
              np.isfinite(np.nanmin(net_noise_mask) == False)):
            # else if all values are nans
            pass
        else:
            # else if there are different finite values
            print("Error! The noise mask has varying values; cannot normalize to 1.")
            sys.exit()

        # at small radii, there is not enough room for a necklace of patches, so just put a dummy blank in
        if (len(other_angles) < 2):
            necklace_2d_array = np.nan*np.ones(np.shape(smoothed_adi_frame))

        # replace zeros in the necklace 2d array with NaNs
        necklace_2d_array[necklace_2d_array == 0] = np.nan
        ## find S/N
        noise_smoothed_full_annulus = np.multiply(smoothed_adi_frame,net_noise_mask)
        comp_ampl = np.multiply(smoothed_adi_frame,comp_mask_inv)
        signal = np.nanmax(comp_ampl)
        if (noise_option == "full_ring"):
            noise = np.nanstd(noise_smoothed_full_annulus)
            noise_frame = noise_smoothed_full_annulus
        elif (noise_option == "necklace"):
            noise = np.nanstd(patch_center_array)
            noise_frame = necklace_2d_array
        s2n = np.divide(signal,noise)

        # append S/N info if first iteration; if N>1 iteration,
        # fill in the nans in the last rows which correspond to that companion
        injection_loc_dict["host_ampl"] = host_ampl
        injection_loc_dict["signal"] = signal
        injection_loc_dict["noise"] = noise
        injection_loc_dict["s2n"] = s2n

        # last step size for fake planet injection
        #injection_loc_dict["last_ampl_step_signed"] = np.nan
        #injection_loc_dict["inject_iteration"] = self.inject_iteration
        #injection_loc_dict["crossover_last_step"] = False

        print("-"*prog_bar_width)
        print("Host star amplitude:")
        print(host_ampl)
        print("Signal:")
        print(signal)
        print("Noise:")
        print(noise)
        print("S/N:")
        print(s2n)
        print("-"*prog_bar_width)

        # append to csv
        injection_loc_df = pd.DataFrame(injection_loc_dict)

        # check if csv file exists; if it does, don't repeat the header
        exists = os.path.isfile(self.csv_record_file_name)
        if (self.injection_iteration == 0):
            # simple append
            injection_loc_df.to_csv(self.csv_record_file_name,
                                    sep = ",",
                                    mode = "a",
                                    header = (not exists))
            print("Appended data to csv ")
        elif (self.injection_iteration > 0):
            # fill in the nans
            to_update_df = pd.read_csv(self.csv_record_file_name, index_col=0)
            to_update_df.reset_index(inplace=True,drop=True)
            #import ipdb; ipdb.set_trace()
            df_this_iteration = to_update_df.where(to_update_df["inject_iteration"] == self.injection_iteration)
            # zero in on row of interest (make the other rows nans)
            df_this_iteration_this_loc = df_this_iteration.where(
                                            np.logical_and(
                                                df_this_iteration["angle_deg"]==injection_loc_dict["angle_deg"][0],
                                                df_this_iteration["rad_asec"]==injection_loc_dict["rad_asec"][0])
                                            )
            # get the index
            df_this_iteration_this_loc_nonan = df_this_iteration_this_loc.dropna(subset=["angle_deg"])
            # insert the new value
            to_update_df.loc[df_this_iteration_this_loc_nonan.index,"signal"] = signal
            to_update_df.loc[df_this_iteration_this_loc_nonan.index,"noise"] = noise
            to_update_df.loc[df_this_iteration_this_loc_nonan.index,"s2n"] = s2n

            # write out (overwrite old file)
            to_update_df.to_csv(self.csv_record_file_name,
                                    sep = ",",
                                    mode = "w",
                                    header = True)
            print("Filled in signal and noise data in csv")

        print(str(self.csv_record_file_name))
        print("-"*prog_bar_width)
        # write out frame as a check
        sn_check_cube = np.zeros((4,np.shape(smoothed_adi_frame)[0],np.shape(smoothed_adi_frame)[1]))
        sn_check_cube[0,:,:] = self.master_frame # the original ADI frame
        sn_check_cube[1,:,:] = smoothed_adi_frame # smoothed frame
        sn_check_cube[2,:,:] = noise_frame # the noise ring (for full_ring mode); or the noise patches (for necklace mode),  note this is blank if there is no room for necklace
        sn_check_cube[3,:,:] = comp_ampl # the area around the companion (be it fake or possibly real)
        fits.writeto(filename = config["data_dirs"]["DIR_S2N_CUBES"] + "sn_check_cube_" + os.path.basename(self.adi_frame_file_name),
                     data = sn_check_cube,
                     overwrite = True)
        print("detection: "+str(datetime.datetime.now())+": Wrote out S/N check cube as \n" + str(config["data_dirs"]["DIR_S2N_CUBES"]) +
              "sn_check_cube_" + os.path.basename(self.adi_frame_file_name))


def main(inject_iteration=None):
    '''
    Detect companions (either fake or in a blind search within science data)
    and calculate S/N.

    INPUT:
    inject_iteration=None: not trying to detect any fake planets
    inject_iteration=int: the number of the iteration for injecting fake planets; new
        entries to be appended in the csv file will have a new iteration number
    '''

    # configuration data
    config = configparser.ConfigParser() # for parsing values in .init file
    config.read("modules/config.ini")

    if (inject_iteration == None):
        injection_iteration_string = "no_fake_planet"
        # the string is not being appended to the path, to avoid breakage
        # with pipeline upstream
        print("detection: No fake planet being sought")
        cookies_centered_06_directory = str(config["data_dirs"]["DIR_CENTERED"])
        print("PLACEHOLDER: NEED TO WRITE IN THE CORRECT READ DIRECTORY AT THIS STEP")
        sys.exit()
    elif (inject_iteration is not None):
        # if we are injecting fake planets, get source images from current iteration
        injection_iteration_string = "inj_iter_" + str(inject_iteration).zfill(4)
        print("detection: Detection of fake planet injection iteration number " + \
            injection_iteration_string)

        # make a list of the images in the ADI directory
        # N.b. fake planet parameters of all zero just indicate there is no fake planet
        hosts_removed_fake_psf_09a_directory = str(config["data_dirs"]["DIR_ADI_W_FAKE_PSFS"]) + \
            injection_iteration_string
        # find all combinations of available fake planet parameters using the file names
        hosts_removed_fake_psf_09a_name_array = list(glob.glob(os.path.join(hosts_removed_fake_psf_09a_directory,
                                                                                "*.fits"))) # list of all files

        # list fake planet parameter patterns from adi_frame_AAAAA_BBBBB_CCCCC_lm_YYMMDD_NNNNNN.fits, where
        # AAAAA is azimuth angle in deg
        # BBBBB is radius in asec
        # CCCCC is contrast
        # examples: adi_frame_270.0_1.3_0.001.fits, adi_frame_270.0_1.1_1e-05.fits
        print(hosts_removed_fake_psf_09a_name_array[0].split("adi_frame_"))
        # the below list may have repeats
        degen_param_list = [i.split("adi_frame_")[1].split(".fits")[0] for i in hosts_removed_fake_psf_09a_name_array]
        param_list = list(frozenset(degen_param_list)) # remove repeats

        # name of file which will record S/N calculations for the INITIAL iteration, for each fake planet parameter
        csv_file_name = config["data_dirs"]["DIR_S2N"] + config["file_names"]["DETECTION_CSV"]
        # name of file which will record S/N calculations for ALL iterations, for each fake planet parameter
        csv_file_name_all_iters = config["data_dirs"]["DIR_S2N"] + config["file_names"]["DETECTION_CSV_ALL_ITER"]

    if (inject_iteration is None):
        print("PLACEHOLDER: NOTHING BEING INJECTED; I JUST WANT TO SEARCH FOR POSSIBLE SIGNAL")
    if (inject_iteration == 0):
        # check if csv file exists for the initial iteration; I want to start with a new one
        exists = os.path.isfile(csv_file_name)
        if exists:
            input("A fake planet detection CSV file already exists, and this is " + \
                "injection iteration number 0! Hit [Enter] to delete CSV and continue.")
            os.remove(csv_file_name)
    if (inject_iteration > 0):
        # read in the pre-existing file and fill in the NaNs in the rows
        # corresponding to this iteration
        pre_existing_data_df = pd.read_csv(csv_file_name_all_iters, index_col=0)
        pre_existing_data_df.reset_index(inplace=True,drop=True)
        csv_file_name = csv_file_name_all_iters # reassign name

    # loop over all fake planet parameter combinations to retrieve ADI frames and look for signal
    for t in range(0,len(param_list)):

        # extract fake planet parameter raw values as ints
        ## ## this may be obsolete, since the values are now read in from the headers
        raw_angle = int(float(param_list[t].split("_")[0]))
        raw_radius = int(float(param_list[t].split("_")[1]))
        raw_contrast = int(float(param_list[t].split("_")[2]))
        print(raw_angle)
        print(raw_radius)
        print(raw_contrast)
        print(param_list[t].split("_")[1])
        print("-----")

        # get physical values
        fake_angle_e_of_n_deg = np.divide(raw_angle,100.)
        fake_radius_asec = np.divide(raw_radius,100.)
        fake_contrast_rel = np.power(10.,-np.divide(raw_contrast,100.)) # scale is relative and linear

        # specify parameters of fake companion
        fake_params_string = param_list[t]

        # initialize and detect
        if (inject_iteration is not None):
            injection_iteration_string = "inj_iter_" + str(inject_iteration).zfill(4)
        else:
            injection_iteration_string = ""
        detection_blind_search = Detection(injection_iteration = inject_iteration,
                                            adi_frame_file_name = config["data_dirs"]["DIR_ADI_W_FAKE_PSFS"] + \
                                                    injection_iteration_string + "/" + \
                                                   "adi_frame_" + fake_params_string + ".fits",
                                                   csv_record_file_name = csv_file_name,
                                                   inject_iteration = inject_iteration)
        detection_blind_search(sci_median_file_name = config["data_dirs"]["DIR_OTHER_FITS"] + \
                                                   config["file_names"]["MEDIAN_SCI_FRAME"],
                                                   noise_option = "full_ring",
                                                   noise_annulus_half_width_pix = 0.5,
                                                   blind_search = False)

        '''
        # STAND-IN FOR A FRAME WHERE THERE IS NO FAKE PLANET, AND I JUST WANT TO MAKE A CRUDE CONTRAST CURVE BASED ON
        # THE NOISE LEVEL IN EACH RING AROUND THE HOST STAR
        # substitute a fake parameter list above this for-loop
        '''

    ###########################################################
    ## ## IMAGES WITHOUT FAKE PLANETS; I.E., ACTUAL SCIENCE

    # MAKE LIST OF ADI FRAMES IN A DIRECTORY (MAY BE JUST 1)

    # DO CROSS-CORRELATION TO FIND MOST LIKELY SPOT WHERE A PLANET EXISTS

    # FIND THE S/N OF THE 'DETECTION'

    # WRITE DATA TO CSV

    # WHILE S/N >2, DO IT AGAIN (WHILE MASKING THE PRECEDING CANDIDATE FOOTPRINTS)
