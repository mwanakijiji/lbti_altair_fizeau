'''
Prepares the data: bad pixel masking, background-subtraction, etc.
## ## This is descended from
## ## find_background_limit.ipynb
## ## subtract stray y illumination gradient parallel.py
## ## make_pca_basis_cube_altair.py
## ## pca_background_subtraction.ipynb
'''

import multiprocessing
import configparser
import glob
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.convolution import convolve, Gaussian1DKernel, interpolate_replace_nans
from modules import *


class DarkSubtSingle:
    '''
    Dark-subtraction of a single raw frame
    '''

    def __init__(self, config_data=config):

        self.config_data = config_data

        # read in dark
        abs_dark_name = str(self.config_data["data_dirs"]["DIR_CALIB_FRAMES"]) + \
          "master_dark.fits"
        self.dark, self.header_dark = fits.getdata(abs_dark_name, 0, header=True)

    def __call__(self, abs_sci_name):
        '''
        Actual subtraction, for a single frame so as to parallelize job

        INPUTS:
        sci_name: science array filename
        '''

        # read in the science frame from raw data directory
        sci, header_sci = fits.getdata(abs_sci_name, 0, header=True)

        # subtract from image; data type should allow negative numbers
        image_dark_subtd = np.subtract(sci, self.dark).astype(np.int32)

        # add a line to the header indicating last reduction step
        header_sci["RED_STEP"] = "dark-subtraction"

        # write file out
        abs_image_dark_subtd_name = str(self.config_data["data_dirs"]["DIR_DARK_SUBTED"] + \
                                        os.path.basename(abs_sci_name))
        fits.writeto(filename=abs_image_dark_subtd_name,
                     data=image_dark_subtd,
                     header=header_sci,
                     overwrite=True)
        print("Writing out dark-subtracted frame " + os.path.basename(abs_sci_name))

        
class FixPixSingle:
    '''
    Interpolates over bad pixels
    '''

    def __init__(self, config_data=config):

        self.config_data = config_data

        # read in bad pixel mask
        # (altair 180507 bpm has convention 0=good, 1=bad)
        abs_badpixmask_name = str(self.config_data["data_dirs"]["DIR_CALIB_FRAMES"]) + \
          "master_bpm.fits"
        self.badpix, self.header_badpix = fits.getdata(abs_badpixmask_name, 0, header=True)

        # (altair 180507 bpm requires a top row)
        self.badpix = np.vstack([self.badpix,np.zeros(np.shape(self.badpix)[1])])

        # turn 1->nan (bad), 0->1 (good) for interpolate_replace_nans
        self.ersatz = np.nan*np.ones(np.shape(self.badpix))
        self.ersatz[self.badpix == 0] = 1.
        self.badpix = self.ersatz # rename
        del self.ersatz

        # define the convolution kernel (normalized by default)
        self.kernel = np.ones((3,3)) # just a patch around the kernel

    def __call__(self, abs_sci_name):
        '''
        Bad pix fixing, for a single frame so as to parallelize job

        INPUTS:
        sci_name: science array filename
        '''

        # read in the science frame from raw data directory
        sci, header_sci = fits.getdata(abs_sci_name, 0, header=True)

        # fix bad pixels
        sci_badnan = np.multiply(sci,self.badpix)
        image_fixpixed = interpolate_replace_nans(array=sci_badnan, kernel=self.kernel)#.astype(np.int32)

        # add a line to the header indicating last reduction step
        header_sci["RED_STEP"] = "bad-pixel-fixed"

        # write file out
        abs_image_fixpixed_name = str(self.config_data["data_dirs"]["DIR_PIXL_CORRTD"] + \
                                        os.path.basename(abs_sci_name))
        fits.writeto(filename=abs_image_fixpixed_name,
                     data=image_fixpixed,
                     header=header_sci,
                     overwrite=True)
        print("Writing out bad-pixel-fixed frame " + os.path.basename(abs_image_fixpixed_name))

    
class RemoveStrayRamp:
    '''
    Removes an additive electronic artifact illumination ramp in y at the top of the 
    LMIRcam readouts. (The ramp has something to do with resetting of the detector while 
    using the 2018A-era electronics; i.e., before MACIE overhaul in summer 2018-- see 
    emails from J. Leisenring and J. Stone, Sept. 5/6 2018)
    '''

    def __init__(self, config_data=config):
        self.config_data = config_data

    def __call__(self, abs_sci_name):
        '''
        Remove stray ramp, for a single frame so as to parallelize job

        INPUTS:
        sci_name: science array filename
        '''

        # read in the science frame from raw data directory
        sci, header_sci = fits.getdata(abs_sci_name, 0, header=True)

        # initialize kernel (for smoothing in y)
        num_pix_y = 5
        gauss_kernel = Gaussian1DKernel(num_pix_y)

        # find the median in x across the whole array as a function of y
        # (N.b. offsets in the channels shouldn't be a problem, since both
        # the ramp and the channel offsets should be additive)
        stray_ramp = np.nanmedian(sci[:,:],axis=1)

        # smooth it
        smoothed_stray_ramp = convolve(stray_ramp, gauss_kernel)

        ## TEST HERE
        #plt.plot(stray_ramp)
        #plt.plot(smoothed_stray_ramp)
        #plt.savefig('junk.png')

        # subtract from the whole array, after tiling across all pixels in x
        # (N.b., again, there will still be residual channel pedestals. These
        # are a mix of sky background and detector bias.)
        image_ramp_subted = np.subtract(sci,
                                        np.tile(smoothed_stray_ramp,(np.shape(sci)[1],1)).T)

        # add a line to the header indicating last reduction step
        header_sci["RED_STEP"] = "ramp_removed"

        # write file out
        abs_image_ramp_subted_name = str(self.config_data["data_dirs"]["DIR_RAMP_REMOVD"] + \
                                        os.path.basename(abs_sci_name))
        fits.writeto(filename=abs_image_ramp_subted_name,
                     data=image_ramp_subted,
                     header=header_sci,
                     overwrite=True)
        print("Writing out ramp-removed-fixed frame " + os.path.basename(abs_image_ramp_subted_name))

        
class PCABackgroundCubeMaker:
    '''
    Generates a PCA cube based on the backgrounds in the science frames.
    N.b. The user has to input a beginning and ending frame number for
    calculating a PCA basis (need to have common filter, integ time, etc.)
    '''
    
    def __init__(self, file_list, n_PCA, config_data = config):
        '''
        INPUTS:
        file_list: list of filenames in the directory
        n_PCA: number of PCA components to save in the cube
        config_data: configuration data, as usual
        '''

        self.file_list = file_list
        self.n_PCA = n_PCA
        self.config_data = config_data

    def __call__(self,
                 start_frame_num,
                 stop_frame_num,
                 quad_choice,
                 indiv_channel=False):
        '''
        Make PCA cube to reconstruct background (PSF is masked)

        INPUTS:
        start_frame_num: starting frame number to use in PCA basis generation
        stop_frame_num: stopping [ditto], inclusive
        quad_choice: quadrant (1 to 4) of the array we are interested in making a background for
        indivChannel: do you want to append PCA components that involve individual channel pedestals?
        '''

        # read in a first file to get the shape
        test_img, header = fits.getdata(self.file_list[0], 0, header=True)
        shape_img = np.shape(test_img)
        
        print("Initializing a PCA cube...")
        training_cube = np.nan*np.ones((stop_frame_num-start_frame_num+1,shape_img[0],shape_img[1]), dtype = np.int64)

        mask_weird = make_first_pass_mask(quad_choice) # make the right mask

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
        fits.writeto('junkmask.fits', mask_weird, overwrite = True)
        training_cube_masked_weird = np.multiply(training_cube,mask_weird)
        del training_cube

        # at this point, test_cube holds the (masked) background frames to be used as a training set

        # find the 2D median across all background arrays, and subtract it from each individual background array
        # (N.b. the region of the PSF itself will look funny, but we'll mask this in due course)
        median_2d_bckgrd = np.nanmedian(training_cube_masked_weird, axis=0)    
        for t in range(0,stop_frame_num-start_frame_num+1):
            training_cube_masked_weird[t,:,:] = np.subtract(training_cube_masked_weird[t,:,:],median_2d_bckgrd)

        # at this point, test_cube holds the background frames which are dark- and 2D median-subtracted 

        ######################
        # write out cube
        cube_file_name = 'junkcube.fits'
        abs_cube_name = str(self.config_data["data_dirs"]["DIR_OTHER_FITS"] + \
                                        os.path.basename(cube_file_name))
        fits.writeto(filename=abs_cube_name,
                     data=training_cube_masked_weird,
                     overwrite=True)
        print("Wrote out background PCA cube for frames " + os.path.basename(abs_cube_name))
        
        
class PCABackgroundSubt:
    '''
    Does a PCA decomposition of a given frame, and subtracts the background
    ## ## N.b. remaining pedestal should be photons alone; how smooth is it?
    '''

    pass

    
class CalcNoise:
    '''
    Finds noise characteristics: where is the background limit? etc.
    '''

    pass

class CookieCutout:
    '''
    Cuts out region around PSF commensurate with the AO control radius
    '''


def main():
    '''
    Carry out the basic data-preparation steps
    '''

    # configuration data
    config = configparser.ConfigParser() # for parsing values in .init file
    config.read("modules/altair_config.ini")

    # multiprocessing instance
    pool = multiprocessing.Pool(ncpu)

    '''
    # make a list of the raw files
    raw_00_directory = str(config["data_dirs"]["DIR_RAW_DATA"])
    raw_00_name_array = list(glob.glob(os.path.join(raw_00_directory, "*.fits")))

    # subtract darks in parallel
    print("Subtracting darks with " + str(ncpu) + " CPUs...")
    do_dark_subt = DarkSubtSingle(config)
    pool.map(do_dark_subt, raw_00_name_array)

    # make a list of the dark-subtracted files
    darksubt_01_directory = str(config["data_dirs"]["DIR_DARK_SUBTED"])
    darksubt_01_name_array = list(glob.glob(os.path.join(darksubt_01_directory, "*.fits")))

    # fix bad pixels in parallel
    print("Fixing bad pixels with " + str(ncpu) + " CPUs...")
    do_fixpix = FixPixSingle(config)
    pool.map(do_fixpix, darksubt_01_name_array)

    # make a list of the bad-pix-fixed files
    fixpixed_02_directory = str(config["data_dirs"]["DIR_PIXL_CORRTD"])
    fixpixed_02_name_array = list(glob.glob(os.path.join(fixpixed_02_directory, "*.fits")))

    # subtract ramps in parallel
    print("Subtracting artifact ramps with " + str(ncpu) + " CPUs...")
    do_ramp_subt = RemoveStrayRamp(config)
    pool.map(do_ramp_subt, fixpixed_02_name_array)
    '''
    
    # make a list of the ramp-removed files
    ramp_subted_03_directory = str(config["data_dirs"]["DIR_RAMP_REMOVD"])
    ramp_subted_03_name_array = list(glob.glob(os.path.join(ramp_subted_03_directory, "*.fits")))

    # generate PCA cubes for backgrounds
    pca_back_maker = PCABackgroundCubeMaker(file_list = ramp_subted_03_name_array,
                                            n_PCA = 3) # create instance
    pca_back_maker(start_frame_num = 9000,
                   stop_frame_num = 9009,
                   quad_choice = 3,
                   indiv_channel = True)
