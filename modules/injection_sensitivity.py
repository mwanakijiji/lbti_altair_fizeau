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
from modules import host_removal, detection

# import the PCA machinery for making backgrounds
from .basic_red import BackgroundPCACubeMaker 

import matplotlib
matplotlib.use('agg') # avoids some crashes when multiprocessing
import matplotlib.pyplot as plt


class FakePlanetInjectorCube:
    '''
    PCA-decompose host star PSF and inject fake planet PSFs,
    based on a grid of fake planet parameters

    RETURNS:
    Cube of non-derotated frames with fake planets injected
    An array of parallactic angles
    An array of integers indicating the frame number (from the original file name)
    '''

    def __init__(self,
                 fake_params,
                 n_PCA,
                 abs_PCA_name,
                 config_data = config,
                 write = False):
        '''
        INPUTS:
        fake_params: parameters of the fake companion
        n_PCA: number of principal components to use
        abs_PCA_name: absolute file name of the PCA cube to reconstruct the host star
                       for making a fake planet (i.e., without saturation effects)
        fake_params_pre_permute: angles (relative to PA up), radii, and amplitudes (normalized to host star) of fake PSFs
                       ex.: fake_params = {"angle_deg_EofN": [0., 60., 120.], "rad_asec": [0.3, 0.4], "ampl_linear_norm": [1., 0.9]}
                       -> all permutations of these parameters will be computed later
        config_data: configuration data, as usual
        write: flag as to whether data product should be written to disk (for checking)
        '''

        self.n_PCA = n_PCA
        self.abs_PCA_name = abs_PCA_name
        self.config_data = config_data
        self.write = write

        # read in the PCA vector cube for this series of frames
        self.pca_basis_cube_unsat, self.header_pca_basis_cube_unsat = fits.getdata(self.abs_PCA_name, 0, header=True)

        # parameters of fake companion
        # N.b. this is a SINGLE vector specifying one set of fake companion characteristics
        self.fake_params = fake_params



    def __call__(self,
                 abs_sci_name_array):
        '''
        Reconstruct and inject, for a list of frames

        INPUTS:

        abs_sci_name_array: array of the absolute path of the science frames into which we want to inject a planet
        '''

        print('-----')
        print(self.fake_params)
        print(self.fake_params["angle_deg_EofN"])

        # read in one frame to get the shape
        test_image = fits.getdata(abs_sci_name_array[0], 0, header=False)

        # initialize cube to hold the frames
        print("Memory error, 0 " + str(len(abs_sci_name_array)))
        print("Memory error, shape " + str(np.shape(test_image)))
        cube_frames = np.nan*np.ones((len(abs_sci_name_array),np.shape(test_image)[0],np.shape(test_image)[1]))
        # initialize the array to hold the parallactic angles (for de-rotation later)
        pa_array = np.nan*np.ones(len(abs_sci_name_array))
        # initialize the array to hold the frame numbers (to define masks to apply over pixel regions to make them
        #    NaNs before taking the median of a cube)
        frame_nums_array = np.ones(len(abs_sci_name_array)).astype(int)

        # loop over frames to inject fake planets in each of them
        for frame_num in range(0,len(abs_sci_name_array)):
            print("---------------")
            print("Injecting a fake planet into cube slice " + str(frame_num))

            # read in the cutout science frames
            sci, header_sci = fits.getdata(abs_sci_name_array[frame_num], 0, header=True)

            # define the mask of this science frame
            ## ## fine-tune this step later!
            mask_weird = np.ones(np.shape(sci))
            no_mask = np.copy(mask_weird) # a non-mask for reconstructing sat PSFs
            mask_weird[sci > 55000] = np.nan # mask saturating region

            ## TEST: WRITE OUT
            #hdu = fits.PrimaryHDU(mask_weird)
            #hdulist = fits.HDUList([hdu])
            #hdu.writeto("junk_mask.fits", clobber=True)
            ## END TEST

            ###########################################
            # PCA-decompose the host star PSF to generate a fake planet
            # (be sure to mask the bad regions)
            # (note no de-rotation of the image here)

            # given this sci frame, retrieve the appropriate PCA frame

            # do the PCA fit of masked host star
            # returns dict: 'pca_vector': the PCA best-fit vector; and 'recon_2d': the 2D reconstructed PSF
            # N.b. PCA reconstruction will be to get an UN-sat PSF; note PCA basis cube involves unsat PSFs
            fit_unsat = fit_pca_star(self.pca_basis_cube_unsat, sci, mask_weird, n_PCA=100)
            if not fit_unsat: # if the dimensions were incompatible, skip this science frame
                print("Incompatible dimensions; skipping this frame...")
                continue

            # get absolute amplitude of the host star
            ampl_host_star = np.max(fit_unsat["recon_2d"])

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

            fake_angle_e_of_n_deg = self.fake_params["angle_deg_EofN"]
            fake_radius_asec = self.fake_params["rad_asec"]
            fake_contrast_rel = self.fake_params["ampl_linear_norm"]

            # now calculate where in (y,x) space the fake planet should be injected in the
            # frame as projected in ALT-AZ mode, BEFORE it is de-rotated
            pos_info = polar_to_xy(pos_info = self.fake_params,
                                   pa = header_sci["LBT_PARA"])
            
            print('pa in header:')
            print(header_sci["LBT_PARA"])
            print('fake angle E of N:')
            print(fake_angle_e_of_n_deg)

            # shift the PSF image to the location of the fake planet
            reconImg_shifted = scipy.ndimage.interpolation.shift(
                fit_unsat["recon_2d"],
                shift = [self.fake_params["y_pix_coord"],
                         self.fake_params["x_pix_coord"]]) # shift in +y,+x convention

            print('fake_params y x')
            print(self.fake_params["y_pix_coord"])
            print(self.fake_params["x_pix_coord"])

            # scale the amplitude of the host star to get the fake planet's amplitude
            reconImg_shifted_ampl = np.multiply(reconImg_shifted,
                                                self.fake_params["ampl_linear_norm"])

            # actually inject it
            image_w_fake_planet = np.add(sci, reconImg_shifted_ampl)

            # add image to cube, add PA to array, and add frame number to array
            cube_frames[frame_num] = image_w_fake_planet
            pa_array[frame_num] = header_sci["LBT_PARA"]
            frame_nums_array[frame_num] = int(os.path.basename(abs_sci_name_array[frame_num]).split("_")[-1].split(".")[0])
                

            ## TEST: WRITE OUT
            #hdu = fits.PrimaryHDU(reconImg_shifted)
            #hdulist = fits.HDUList([hdu])
            #hdu.writeto("junk_fake.fits", clobber=True)
            ## END TEST

        # if writing to disk for checking
        if self.write:

            hdr = fits.Header()
            hdr["ANGEOFN"] = self.fake_params["angle_deg_EofN"]
            hdr["RADASEC"] = self.fake_params["rad_asec"]
            hdr["AMPLIN"] = self.fake_params["ampl_linear_norm"]

            file_name = self.config_data["data_dirs"]["DIR_OTHER_FITS"] + "fake_planet_injected_cube_" + \
              str(self.fake_params["angle_deg_EofN"]) + "_" + str(self.fake_params["rad_asec"]) + \
              "_" + str(self.fake_params["ampl_linear_norm"]) + ".fits"
            fits.writeto(filename = file_name,
                         data = cube_frames,
                         header = hdr,
                         overwrite = True)
            print("Wrote fake-planet-injected cube to disk as " + file_name)
        
        print("Array of PA")
        print(pa_array)

        # return cube of frames and array of PAs
        return cube_frames, pa_array, frame_nums_array


def inject_remove_adi(this_param_combo):
    '''
    To parallelize a serial operation across cores, I need to define this function that goes through
    the fake planet injection, host star removal, and ADI steps for a given combination of fake planet parameters
    '''

    time_start = time.time()

    # make a list of the centered cookie cutout files
    cookies_centered_06_directory = str(config["data_dirs"]["DIR_CENTERED"])
    
    '''
    COMMENTED OUT TO JUST USE FRAMES FROM SEQUENCES A AND D
    cookies_centered_06_name_array = list(glob.glob(os.path.join(cookies_centered_06_directory, "*.fits")))
    '''

    '''
    THE BELOW COMMENTED OUT TO AVOID MEMORY ERRORS
    cookies_centered_06_name_array = list(glob.glob(os.path.join(cookies_centered_06_directory, "*_004*.fits")))
    cookies_centered_06_name_array = list(glob.glob(os.path.join(cookies_centered_06_directory, "*_005[012345]*.fits")))
    cookies_centered_06_name_array.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_0058[3456789]*.fits")))
    cookies_centered_06_name_array.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_0059*.fits")))
    cookies_centered_06_name_array.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_006[012]*.fits")))
    cookies_centered_06_name_array.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_00[89]*.fits")))
    cookies_centered_06_name_array.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_010[0123456]*.fits")))
    cookies_centered_06_name_array.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_010[89]*.fits")))
    cookies_centered_06_name_array.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_011*.fits")))
    '''

    # THIS LINE IS AN ERSATZ FOR TESTING ONLY; WILL NEED TO INCLUDE MORE AND BETTER FRAMES NEXT
    cookies_centered_06_name_array = list(glob.glob(os.path.join(cookies_centered_06_directory, "*_00902*.fits")))

    print("number of frames being considered for ADI: " + str(len(cookies_centered_06_name_array)))
    

    ## Inject a fake psf in each science frame, return a cube of non-derotated, non-host-star-subtracted frames
    print("-------------------------------------------------")
    print("Injecting fake planet corresponding to parameter")
    print(this_param_combo)

    # instantiate fake planet injection
    inject_fake_psfs = FakePlanetInjectorCube(fake_params = this_param_combo,
                                          n_PCA = 100,
                                          abs_PCA_name = config["data_dirs"]["DIR_OTHER_FITS"] \
                                          + "pca_cubes_psfs/" \
                                          + "psf_PCA_vector_cookie_seqStart_004259_seqStop_005600.fits",
                                          write = False)

    # call fake planet injection
    injected_fake_psfs_cube, pas_array, frame_array_0 = inject_fake_psfs(cookies_centered_06_name_array)

    # fyi
    print("Frames into which we will inject fake planets: ")
    print(frame_nums_array)

    # instantiate removal of host star from each frame in the cube
    remove_hosts = host_removal.HostRemovalCube(fake_params = this_param_combo,
                                                    cube_frames = injected_fake_psfs_cube,
                                                    n_PCA = 100,
                                                    outdir = config["data_dirs"]["DIR_FAKE_PSFS_HOST_REMOVED"],
                                                    abs_PCA_name = config["data_dirs"]["DIR_OTHER_FITS"] \
                                                          + "pca_cubes_psfs/" \
                                                          + "psf_PCA_vector_cookie_seqStart_004259_seqStop_005600.fits",
                                                    frame_array = frame_array_0,
                                                    write = False)

    # call and return cube of host-removed frames
    removed_hosts_cub, frame_array_1 = remove_hosts()

    # instantiate derotation, ADI, sensitivity determination
    median_instance = detection.MedianCube(fake_params = this_param_combo,
                                               host_subt_cube = removed_hosts_cube,
                                               pa_array = pas_array,
                                               frame_array = frame_array_1,
                                               write_cube = True)

    fake_params_string = "STANDIN"

    # call derotation, ADI, sensitivity determination
    make_median = median_instance(fake_planet = True)

    elapsed_time = np.subtract(time.time(), time_start)

    print("----------------------------------------------------------------")
    print("Completed one fake planet parameter configuration")
    print("Elapsed time (sec): ")
    print(str(int(elapsed_time)))


def main():
    '''
    Make grid of fake planet parameters, and inject fake planets
    '''

    # configuration data
    config = configparser.ConfigParser() # for parsing values in .init file
    config.read("modules/config.ini")

    # fake planet injection parameters
    fake_params_pre_permute = {"angle_deg_EofN": [270.],
                               "rad_asec": [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9],
                               "ampl_linear_norm": [1e-3, 1e-4, 1e-5]}

    ## ## generalize the retrieved PCA vector cube as function of science frame range later!

    # permutate values of fake planet parameters to get all possible combinations
    keys, values = zip(*fake_params_pre_permute.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    # convert to dataframe
    experiment_vector = pd.DataFrame(experiments)
    # convert radii in asec to pixels
    ## ## functionality of polar_to_xy will need to be checked, since I changed the convention in the init
    ## ## file to be deg E of N, and in asec
    experiment_vector["rad_pix"] = np.divide(experiment_vector["rad_asec"],
                                             np.float(config["instrum_params"]["LMIR_PS"]))

    # map inject_remove_adi() over all cores, over single combinations of fake planet parameters
    pool = multiprocessing.Pool(ncpu)

    # create list of dictionaries of fake planet parameters
    # (one dictionary is fed to each core at a time)
    param_dict_list = []
    for k in range(0,len(experiment_vector)):
        param_dict_list.append(experiment_vector.iloc[k].to_dict())
    pool.map(inject_remove_adi, param_dict_list)



