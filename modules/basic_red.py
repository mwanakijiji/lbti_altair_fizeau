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
from astropy.io import fits
from astropy.convolution import interpolate_replace_nans
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

        pass

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

class PCABackgroundDecomp:
        '''
        Generates a PCA cube based on the backgrounds in the science frames.
        '''

        pass

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

    # make a list of the raw files
    raw_00_directory = str(config["data_dirs"]["DIR_RAW_DATA"])
    raw_00_name_array = list(glob.glob(os.path.join(raw_00_directory, "*.fits")))

    # subtract darks in parallel
    print("Subtracting darks with " + str(ncpu) + " CPUs...")
    do_dark_subt = DarkSubtSingle(config)
    pool = multiprocessing.Pool(ncpu)
    pool.map(do_dark_subt, raw_00_name_array)

    # make a list of the dark-subtracted files
    darksubt_01_directory = str(config["data_dirs"]["DIR_DARK_SUBTED"])
    darksubt_01_name_array = list(glob.glob(os.path.join(darksubt_01_directory, "*.fits")))

    # fix bad pixels in parallel
    print("Fixing bad pixels with " + str(ncpu) + " CPUs...")
    do_fixpix = FixPixSingle(config)
    pool = multiprocessing.Pool(ncpu)
    pool.map(do_fixpix, darksubt_01_name_array)

    # make a list of the bad-pix-fixed files
    fixpixed_02_directory = str(config["data_dirs"]["DIR_PIXL_CORRTD"])
    fixpixed_02_name_array = list(glob.glob(os.path.join(fixpixed_02_directory, "*.fits")))
