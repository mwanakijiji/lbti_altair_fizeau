import multiprocessing
import configparser
import glob
import time
import itertools
import pandas as pd
import pickle
import math
from astropy.io import fits
from astropy.convolution import convolve, Gaussian1DKernel, interpolate_replace_nans
from astropy.modeling import models, fitting
from modules import *

# import the PCA machinery for making backgrounds
from .basic_red import BackgroundPCACubeMaker 

import matplotlib
matplotlib.use('agg') # avoids some crashes when multiprocessing
import matplotlib.pyplot as plt


class FakePlanetInjector:
    '''
    PCA-decompose host star PSF and inject fake planet PSFs,
    based on a grid of fake planet parameters

    '''

    def __init__(self,
                 fake_params_pre_permute,
                 n_PCA,
                 abs_PCA_name,
                 config_data = config):
        '''
        INPUTS:
        n_PCA: number of principal components to use
        abs_PCA_name: absolute file name of the PCA cube to reconstruct the host star
                       for making a fake planet (i.e., without saturation effects)
        fake_params_pre_permute: angles (relative to PA up), radii, and amplitudes (normalized to host star) of fake PSFs
                       ex.: fake_params = {"angle_deg": [0., 60., 120.], "rad_asec": [0.3, 0.4], "ampl_linear_norm": [1., 0.9]}
                       -> all permutations of these parameters will be computed later
        config_data: configuration data, as usual
        '''

        self.n_PCA = n_PCA
        self.abs_PCA_name = abs_PCA_name
        self.config_data = config_data

        # read in the PCA vector cube for this series of frames
        self.pca_basis_cube_unsat, self.header_pca_basis_cube_unsat = fits.getdata(self.abs_PCA_name, 0, header=True)

        # permutate values of fake planet parameters to get all possible combinations
        keys, values = zip(*fake_params_pre_permute.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # convert to dataframe
        self.experiment_vector = pd.DataFrame(experiments)
        
        # convert radii in asec to pixels
        ## ## functionality of polar_to_xy will need to be checked, since I changed the convention in the init
        ## ## file to be deg E of N, and in asec
        self.experiment_vector["rad_pix"] = np.divide(self.experiment_vector["rad_asec"],
                                                      np.float(self.config_data["instrum_params"]["LMIR_PS"]))
        self.pos_info = polar_to_xy(self.experiment_vector)

        
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
        sci, header_sci = fits.getdata(abs_sci_name, 0, header=True)

        # define the mask of this science frame
        ## ## fine-tune this step later!
        mask_weird = np.ones(np.shape(sci))
        no_mask = np.copy(mask_weird) # a non-mask for reconstructing sat PSFs
        mask_weird[sci > 55000] = np.nan # mask saturating region

        ## TEST: WRITE OUT
        hdu = fits.PrimaryHDU(mask_weird)
        hdulist = fits.HDUList([hdu])
        hdu.writeto("junk_mask.fits", clobber=True)
        ## END TEST
        
        ###########################################
        # PCA-decompose the host star PSF
        # (be sure to mask the bad regions)
        # (note no de-rotation of the image here)

        # given this sci frame, retrieve the appropriate PCA frame

        # do the PCA fit of masked host star
        # returns dict: 'pca_vector': the PCA best-fit vector; and 'recon_2d': the 2D reconstructed PSF
        # N.b. PCA reconstruction will be to get an UN-sat PSF; note PCA basis cube involves unsat PSFs
        fit_unsat = fit_pca_star(self.pca_basis_cube_unsat, sci, mask_weird, n_PCA=100)

        # get absolute amplitude of the host star
        ampl_host_star = np.max(fit_unsat["recon_2d"])

        ## ## is this line necessary?
        #experiment_vector["ampl_linear_abs"] = np.multiply(self.experiment_vector["ampl_linear_norm"],
        #                                                        np.max(fit_unsat["recon_2d"])) # maybe should do this after smoothing?

        # pickle the PCA vector
        pickle_stuff = {"pca_cube_file_name": self.abs_PCA_name,
                        "pca_vector": fit_unsat["pca_vector"],
                        "recons_2d_psf_unsat": fit_unsat["recon_2d"],
                        "sci_image_name": abs_sci_name}
        print(pickle_stuff)
        pca_fit_pickle_write_name = str(self.config_data["data_dirs"]["DIR_PICKLE"]) \
          + "pickle_pca_psf_info_" + str(os.path.basename(abs_sci_name).split(".")[0]) + ".pkl"
        print(pca_fit_pickle_write_name)
        with open(pca_fit_pickle_write_name, "wb") as f:
            pickle.dump(pickle_stuff, f)
        
        ###########################################
        # inject the fake planet
        # (parameters are:
        # [0]: angle east (in deg) from true north (i.e., after image derotation)
        # [1]: radius (in asec)
        # [2]: contrast ratio (A_star/A_planet, where A_star
        #            is from the PCA_reconstructed star, since the
        #            empirical stellar PSF will have saturated/nonlinear regions)

        ## ## (see inject_fake_planets_test1.ipynb)
        # loop over all elements in the parameter vector
        # (N.b. each element represents one fake planet)
        for elem_num in range(0,len(self.experiment_vector)):

            print(self.experiment_vector)
            
            fake_angle_e_of_n_deg = self.experiment_vector["angle_deg"][elem_num]
            fake_radius_asec = self.experiment_vector["rad_asec"][elem_num]
            fake_contrast_rel = self.experiment_vector["ampl_linear_norm"][elem_num]

            # from these parameters, make strings for the filename
            str_fake_angle_e_of_n_deg = str("{:0>5d}".format(int(100*fake_angle_e_of_n_deg))) # 10.5 [deg] -> "01050" etc.
            str_fake_radius_asec = str("{:0>5d}".format(int(100*fake_radius_asec))) # 5.05 [asec] -> "00505" etc. 
            str_fake_contrast_rel = str("{:0>5d}".format(int(100*np.abs(math.log10(fake_contrast_rel))))) # 10^(-4) -> "00400" etc.

            # find the injection angle, given the PA of the image
            # (i.e., add angle east of true North, and parallactic angle; don't de-rotate the image)
            angle_static_frame_injection = np.add(fake_angle_e_of_n_deg,header_sci["LBT_PARA"])
                
            # shift the image to the right location
            reconImg_shifted = scipy.ndimage.interpolation.shift(
                fit_unsat["recon_2d"],
                shift = [self.experiment_vector["y_pix_coord"][elem_num],
                         self.experiment_vector["x_pix_coord"][elem_num]]) # shift in +y,+x convention
                         
            # scale the amplitude of the host star to get the fake planet's amplitude
            reconImg_shifted_ampl = np.multiply(reconImg_shifted,
                                                self.experiment_vector["ampl_linear_norm"][elem_num])

            # actually inject it
            image_w_fake_planet = np.add(sci, reconImg_shifted_ampl)

            ## TEST: WRITE OUT
            #hdu = fits.PrimaryHDU(training_cube)
            #hdulist = fits.HDUList([hdu])
            #hdu.writeto("junk.fits", clobber=True)
            ## END TEST
                    
            # add info to the header indicating last reduction step, and fake PSF parameters and PCA info
            header_sci["RED_STEP"] = "fake_planet_injection"
            header_sci.comments["RED_STEP"] = "Last reduction step performed"
            header_sci["FAKEAEON"] = fake_angle_e_of_n_deg
            header_sci.comments["FAKEAEON"] = "Fake companion angle after de-rotation (deg E of N)"
            header_sci["FAKERADA"] = fake_radius_asec
            header_sci.comments["FAKERADA"] = "Fake companion radius from star (asec)"
            header_sci["FAKECREL"] = fake_contrast_rel
            header_sci.comments["FAKECREL"] = "Fake companion contrast (relative, and normalized to 1)"
            header_sci["FAKECL10"] = math.log10(fake_contrast_rel)
            header_sci.comments["FAKECL10"] = "Fake companion log10 contrast (relative)"

            # PCA vector file with which the host star was decomposed
            #header_sci["FAKE_PLANET_PCA_CUBE_FILE"] = os.path.basename(self.abs_PCA_name)

            # PCA spectrum of the host star using the above PCA vector, with which the
            # fake planet is injected
            #header_sci["FAKE_PLANET_PCA_SPECTRUM"] = 

            # write FITS file out, with fake planet params in file name and header
            ## ## do I actually want to write out a separate FITS file for each fake planet?
            abs_image_w_fake_planet_name = str(self.config_data["data_dirs"]["DIR_FAKE_PSFS"] + \
                                                 "fake_planet_" + \
                                                 str_fake_angle_e_of_n_deg + "_" + \
                                                 str_fake_radius_asec + "_" + \
                                                 str_fake_contrast_rel + "_" + \
                                                 os.path.basename(abs_sci_name))
            fits.writeto(filename = abs_image_w_fake_planet_name,
                     data = image_w_fake_planet,
                     header = header_sci,
                     overwrite = True)
            print("Writing out fake-planet-injected frame " + os.path.basename(abs_image_w_fake_planet_name))
        
        


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

    # fake planet injection parameters
    fake_params_pre_permute = {"angle_deg": [0., 60., 120.],
                               "rad_asec": [0.3, 0.4],
                               "ampl_linear_norm": [1., 0.9]}

    # initialize and parallelize
    ## ## generalize the retrieved PCA vector cube as function of science frame range later!
    inject_fake_psfs = FakePlanetInjector(fake_params_pre_permute,
                                          n_PCA = 100,
                                          abs_PCA_name = config["data_dirs"]["DIR_OTHER_FITS"] \
                                          + "pca_cubes_psfs/" \
                                          + "psf_PCA_vector_cookie_seqStart_004259_seqStop_005600.fits")

    pool.map(inject_fake_psfs, cookies_centered_06_name_array)
