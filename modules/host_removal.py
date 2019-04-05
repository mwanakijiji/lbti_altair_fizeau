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


class HostRemoval:
    '''
    PCA-decompose a saturated host star PSF and remove it
    '''

    def __init__(self,
                 n_PCA,
                 outdir,
                 abs_PCA_name,
                 config_data = config):
        '''
        INPUTS:
        n_PCA: number of principal components to use
        outdir: directory to deposit the host-subtracted images in (this has to be
                       defined at the function call because the images may or may not
                       contain fake planet PSFs, and I want to keep them separate)
        abs_PCA_name: absolute file name of the PCA cube to reconstruct the host star
                       for making a fake planet (i.e., without saturation effects)
        config_data: configuration data, as usual
        '''

        self.n_PCA = n_PCA
        self.outdir = outdir
        self.abs_PCA_name = abs_PCA_name
        self.config_data = config_data

        # read in the PCA vector cube for this series of frames
        # (note the PCA needs to correspond to saturated PSFs, since I am subtracting
        # saturated PSFs away)
        self.pca_basis_cube_sat, self.header_pca_basis_cube_sat = fits.getdata(self.abs_PCA_name, 0, header=True)

        
        ##########
        

    def __call__(self,
                 abs_sci_name):
        '''
        Reconstruct and inject, for a single frame so as to parallelize the job

        INPUTS:

        abs_sci_name: the absolute path of the science frame into which we want to inject a planet
        '''

        print(abs_sci_name)
        
        # read in the cutout science frame
        # (there should be no masking of this frame downstream)
        sci, header_sci = fits.getdata(abs_sci_name, 0, header=True)

        # define the mask of this science frame
        ## ## fine-tune this step later!
        mask_weird = np.ones(np.shape(sci))
        no_mask = np.copy(mask_weird) # a non-mask for reconstructing saturated PSFs
        #mask_weird[sci > 1e8] = np.nan # mask saturating region

        ## TEST: WRITE OUT
        #hdu = fits.PrimaryHDU(mask_weird)
        #hdulist = fits.HDUList([hdu])
        #hdu.writeto("junk_mask.fits", clobber=True)
        ## END TEST
        
        ###########################################
        # PCA-decompose the host star PSF
        # (note no de-rotation of the image here)
        
        # do the PCA fit of masked host star
        # returns dict: 'pca_vector': the PCA best-fit vector; and 'recon_2d': the 2D reconstructed PSF
        # N.b. PCA reconstruction will be to get an UN-sat PSF; note PCA basis cube involves unsat PSFs
        fit_unsat = fit_pca_star(self.pca_basis_cube_unsat, sci, no_mask, n_PCA=100)

        # subtract the PCA-reconstructed host star
        image_host_removed = np.subtract(sci,fit_unsat)

        # pickle the PCA vector
        pickle_stuff = {"pca_cube_file_name": self.abs_PCA_name,
                        "pca_vector": fit_unsat["pca_vector"],
                        "recons_2d_psf_unsat": fit_unsat["recon_2d"],
                        "sci_image_name": abs_sci_name}
        print(pickle_stuff)
        pca_fit_pickle_write_name = str(self.config_data["data_dirs"]["DIR_PICKLE"]) \
          + "pickle_pca_sat_psf_info_" + str(os.path.basename(abs_sci_name).split(".")[0]) + ".pkl"
        print(pca_fit_pickle_write_name)
        with open(pca_fit_pickle_write_name, "wb") as f:
            pickle.dump(pickle_stuff, f)
                    
        # add info to the header indicating last reduction step, and PCA info
        header_sci["RED_STEP"] = "host_removed"

        # write FITS file out, with fake planet params in file name
        ## ## do I actually want to write out a separate FITS file for each fake planet?
        abs_image_host_removed_name = str(self.outdir + os.path.basename(abs_sci_name))
        fits.writeto(filename = abs_image_host_removed_name,
                     data = image_host_removed,
                     header = header_sci,
                     overwrite = True)
        print("Writing out host_removed frame " + os.path.basename(abs_sci_name))
        
        


def main():
    '''
    Reconstruct and subtract host star PSFs from images
    '''

    # configuration data
    config = configparser.ConfigParser() # for parsing values in .init file
    config.read("modules/config.ini")

    # multiprocessing instance
    pool = multiprocessing.Pool(ncpu)
    
    # make a list of the images WITH fake planets
    fake_planet_frames_07_directory = str(config["data_dirs"]["DIR_FAKE_PSFS"])
    fake_planet_frames_07_name_array = list(glob.glob(os.path.join(fake_planet_frames_07_directory, "*.fits")))

    # make a list of the images WITHOUT fake planets
    # (these are just the centered frames)
    cookies_centered_06_directory = str(config["data_dirs"]["DIR_CENTERED"])
    cookies_centered_06_name_array = list(glob.glob(os.path.join(cookies_centered_06_directory, "*.fits"))) 

    # initialize and parallelize
    ## ## generalize the retrieved PCA vector cube as function of science frame range later!
    host_removal_fake_planets = HostRemoval(n_PCA = 100,
                                            outdir = config["data_dirs"]["DIR_FAKE_PSFS_HOST_REMOVED"], \
                                            abs_PCA_name = config["data_dirs"]["DIR_OTHER_FITS"] \
                                            + "pca_cubes_psfs/" \
                                            + "psf_PCA_vector_cookie_seqStart_004900_seqStop_004919.fits")
    host_removal_no_fake_planets = HostRemoval(n_PCA = 100,
                                            outdir = config["data_dirs"]["DIR_NO_FAKE_PSFS_HOST_REMOVED"], \
                                            abs_PCA_name = config["data_dirs"]["DIR_OTHER_FITS"] \
                                            + "pca_cubes_psfs/" \
                                            + "psf_PCA_vector_cookie_seqStart_004900_seqStop_004919.fits")
                               
    # remove the host from the frames WITH fake planets
    host_removal_fake_planets(fake_planet_frames_07_name_array[0])
    #pool.map(host_removal, fake_planet_frames_07_name_array[0:6])

    # remove the host from the frames WITHOUT fake planets
    host_removal_no_fake_planets(cookies_centered_06_name_array[0])
    #pool.map(host_removal, cookies_centered_06_name_array[0:6])
