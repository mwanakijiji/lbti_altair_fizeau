import multiprocessing
import configparser
import glob
import time
import itertools
from astropy.io import fits
from astropy.convolution import convolve, Gaussian1DKernel, interpolate_replace_nans
from astropy.modeling import models, fitting
from modules import *

# import the PCA machinery for making backgrounds
from .basic_red import BackgroundPCACubeMaker 

import matplotlib
matplotlib.use('agg') # avoids some crashes when multiprocessing
import matplotlib.pyplot as plt


def fit_pca_star(pca_cube, sciImg, mask_weird, n_PCA):
    '''
    INPUTS:
    pca_cube: cube of PCA components
    img_string: full path name of the science image
    sciImg: the science image
    n_PCA: number of PCA components
    
    RETURNS:
    pca spectrum: spectrum of PCA vector amplitudes
    reconstructed PSF: host star PSF as reconstructed with N PCA vector components
    '''
    
    # apply mask over weird regions to PCA cube
    pca_cube_masked = np.multiply(pca_cube,mask_weird)

    # apply mask over weird detector regions to science image
    sciImg_psf_masked = np.multiply(sciImg,mask_weird)
            
    ## PCA-decompose
        
    # flatten the science array and PCA cube 
    pca_not_masked_1ds = np.reshape(pca_cube,(np.shape(pca_cube)[0],np.shape(pca_cube)[1]*np.shape(pca_cube)[2]))
    sci_masked_1d = np.reshape(sciImg_psf_masked,(np.shape(sciImg_psf_masked)[0]*np.shape(sciImg_psf_masked)[1]))
    pca_masked_1ds = np.reshape(pca_cube_masked,(np.shape(pca_cube_masked)[0],np.shape(pca_cube_masked)[1]*np.shape(pca_cube_masked)[2]))
    
    ## remove nans from the linear algebra
        
    # indices of finite elements over a single flattened frame
    idx = np.logical_and(np.isfinite(pca_masked_1ds[0,:]), np.isfinite(sci_masked_1d)) 
        
    # reconstitute only the finite elements together in another PCA cube and a science image
    pca_masked_1ds_noNaN = np.nan*np.ones((len(pca_masked_1ds[:,0]),np.sum(idx))) # initialize array with slices the length of number of finite elements
    for t in range(0,len(pca_masked_1ds[:,0])): # for each PCA component, populate the arrays without nans with the finite elements
        pca_masked_1ds_noNaN[t,:] = pca_masked_1ds[t,idx]
    sci_masked_1d_noNaN = np.array(1,np.sum(idx)) # science frame
    sci_masked_1d_noNaN = sci_masked_1d[idx] 
        
    # the vector of component amplitudes
    soln_vector = np.linalg.lstsq(pca_masked_1ds_noNaN[0:n_PCA,:].T, sci_masked_1d_noNaN)
        
    # reconstruct the background based on that vector
    # note that the PCA components WITHOUT masking of the PSF location is being
    # used to reconstruct the background
    recon_2d = np.dot(pca_cube[0:n_PCA,:,:].T, soln_vector[0]).T
    
    d = {'pca_vector': soln_vector[0], 'recon_2d': recon_2d}
    
    return d


def polar_to_xy(pos_info):
    '''
    Converts polar vectors (radians, pix) to xy vectors (pix, pix)
    '''
    pos_info["x_pix_coord"] = np.multiply(pos_info["rad_pix"],np.cos(np.multiply(pos_info["angle_deg"],np.pi/180.)))
    pos_info["y_pix_coord"] = np.multiply(pos_info["rad_pix"],np.sin(np.multiply(pos_info["angle_deg"],np.pi/180.)))
    
    return pos_info


class FakePlanetInjector:
    '''
    PCA-decompose host star PSF and inject fake planet PSFs,
    based on a grid of fake planet parameters

    '''

    def __init__(self,
                 file_list,
                 n_PCA,
                 abs_PCA_name,
                 fake_params_pre_permute,
                 config_data = config):
        '''
        INPUTS:
        file_list: list of ALL filenames in the directory
        n_PCA: number of principal components to use
        abs_PCA_name: absolute file name of the PCA cube to reconstruct the host star
                       for making a fake planet (i.e., without saturation effects)
        fake_params_pre_permute: angles (relative to PA up), radii, and amplitudes (normalized to host star) of fake PSFs
                       ex.: fake_params = {"angle_deg": [0., 60., 120.], "rad_asec": [0.3, 0.4], "ampl_linear_norm": [1., 0.9]}
                       -> all permutations of these parameters will be computed later
        config_data: configuration data, as usual
        '''

        self.file_list = file_list
        self.n_PCA = n_PCA
        self.abs_PCA_name = abs_PCA_name
        self.config_data = config_data

        # read in the PCA vector cube for this series of frames
        self.pca_cube_nosat, self.header_pca_cube_nosat = fits.getdata(self.abs_PCA_name, 0, header=True)

        # permutate values to get all possible parameter combinations
        keys, values = zip(*fake_params_pre_permute.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # convert to dataframe
        self.experiment_vector = pd.DataFrame(experiments)
        
        # convert radii in asec to pixels
        self.experiment_vector["rad_pix"] = np.divide(self.experiment_vector["rad_asec"],lmir_ps)
        self.experiment_vector["ampl_linear_abs"] = np.multiply(self.experiment_vector["ampl_linear_norm"],
                                                   np.max(fit_sat["recon_2d"])) # maybe should do this after smoothing?

        self.pos_info = polar_to_xy(self.experiment_vector)
                                                   
        self.ampl_host_star = np.max(fit_sat["recon_2d"]) # FYI for now

        
        ##########
        

    def __call__(self,
                 abs_sci_name):
        '''
        Reconstruct and inject, for a single frame so as to parallelize the job

        INPUTS:

        abs_sci_name: the absolute path of the science frame into which we want to inject a planet
        '''

        # read in the cutout science frame
        sci, header_sci = fits.getdata(abs_sci_name, 0, header=True)

        # define the mask of this science frame
        ## ## fine-tune this step later!
        mask_weird = np.ones(np.shape(sci))
        no_mask = np.copy(mask_weird) # a non-mask for reconstructing sat PSFs
        mask_weird[sci > 35000] = np.nan # mask saturating region

        ###########################################
        # PCA-decompose the host star PSF
        # (be sure to mask the bad regions)
        # (note no de-rotation of the image here)

        # given this sci frame, retrieve the appropriate PCA frame

        # do the PCA fit of masked host star
        # returns: the PCA best-fit vector, and the 2D reconstructed PSF
        # N.b. PCA reconstruction will be to get an UN-sat PSF; note PCA basis cube involves unsat PSFs
        fit_unsat = fit_pca_star(self.pca_basis_cube_unsat, sci, mask_weird, n_PCA=100)
        
        ###########################################
        # inject the fake planet
        # (parameters are:
        # [0]: angle east (in deg) from true north (i.e., after image derotation)
        # [1]: radius (in asec)
        # [2]: contrast ratio (A_star/A_planet, where A_star
        #            is from the PCA_reconstructed star, since the
        #            empirical stellar PSF will have saturated/nonlinear regions)

        # from these parameters, make strings for the filename
        str_fake_angle_e_of_n = str("{:0>5d}".format(int(100*fake_angle_e_of_n))) # 10.5 [deg] -> "01050" etc.
        str_fake_radius = str("{:0>5d}".format(int(100*fake_radius))) # 5.05 [asec] -> "00505" etc. 
        str_fake_contrast = str("{:0>5d}".format(int(100*np.a7UAZ bs(math.log10(fake_contrast))))) # 10^(-4) -> "00400" etc.

        # find the injection angle, given the PA of the image
        # (i.e., add angle east of true North, and parallactic angle; don't de-rotate the image)
        angle_static_frame_injection = np.add(fake_angle_e_of_n,header_sci["LBT_PARA"])

        ## inject the planet at the right position, amplitude (SEE inject_fake_planets_test1.ipynb)

        # loop over all elements in the parameter vector
        # (N.b. each element represents one fake planet)
        for elem_num in range(0,len(self.experiment_vector)):
            # shift the image to the right location
            reconImg_shifted = scipy.ndimage.interpolation.shift(
                fit_unsat["recon_2d"],
                shift = [self.experiment_vector["y_pix_coord"][elem_num],
                         self.experiment_vector["x_pix_coord"][elem_num]]) # shift in +y,+x convention
            # multiply it to get the right amplitude
            reconImg_shifted_ampl = np.multiply(reconImg_shifted,
                                                self.experiment_vector["ampl_linear_norm"][elem_num])

            # actually inject it
            image_w_fake_planet = np.add(reconImg_shifted, ampl_norm*reconImg_shifted)

            # write it out
            ## ## do I actually want to write out a separate FITS file for each fake planet?

            ## ## CURRENT LOCATION
        
            # add info to the header indicating last reduction step, and PCA info
            header_sci["RED_STEP"] = "fake_planet_injection"

            # PCA vector file with which the host star was decomposed
            #header_sci["FAKE_PLANET_PCA_CUBE_FILE"] =

            # PCA spectrum of the host star using the above PCA vector, with which the
            # fake planet is injected
            #header_sci["FAKE_PLANET_PCA_SPECTRUM"] =

            # write FITS file out, with fake planet params in file name
            abs_image_cookie_centered_name = str(self.config_data["data_dirs"]["DIR_CENTERED"] + \
                                             str_fake_angle_e_of_n + "_" + \
                                             str_fake_radius + "_" + \
                                             str_fake_contrast + "_" + \
                                             os.path.basename(abs_sci_name))
            fits.writeto(filename=abs_image_cookie_centered_name,
                     data=sci_shifted,
                     header=header_sci,
                     overwrite=True)
            print("Writing out fake-planet-injected frame " + os.path.basename(abs_sci_name))
        
        


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

    # 
    inject_fake_psfs = FakePlanetInjector()
    pool.map(inject_fake_psfs, cookies_06_name_array) 
