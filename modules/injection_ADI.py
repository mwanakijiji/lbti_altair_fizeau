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

class JustPutIntoCube:
    '''
    Just put centered science frames into a cube, and return it along with info for derotation
    (No host star subtraction is done yet)

    RETURNS:
    Cube of non-derotated frames WITHOUT any fake planets injected
    An array of parallactic angles
    An array of integers indicating the frame number (from the original file name)
    '''

    def __init__(self,
                 fake_params,
                 test_PCA_vector_name,
                 config_data = config,
                 write = False):
        '''
        INPUTS:
        fake_params: parameters of a fake planet (just zeros if none)
        test_PCA_vector_name: absolute file name of the PCA cube to reconstruct the host star
                       (pay attention to the filter combination and saturation status)
        config_data: configuration data, as usual
        write: flag as to whether data product should be written to disk (for checking)
        '''

        self.fake_params = fake_params
        self.test_PCA_vector_name = test_PCA_vector_name
        self.config_data = config_data
        self.write = write

        # read in a test PCA vector cube for this series of frames
        # (this is just for checking if reconstruction is possible)
        self.pca_star_basis_cube, self.header_pca_basis_cube = fits.getdata(self.test_PCA_vector_name,
                                                                            0, header=True)



    def __call__(self,
                 abs_sci_name_array,
                 saved_cube_basename):
        '''
        INPUTS:

        abs_sci_name_array: array of the absolute path of the science frames into which we want to inject a planet
        saved_cube_basename: string for the filename of the cube to be saved
        '''

        # read in one frame to get the shape
        test_image = fits.getdata(abs_sci_name_array[0], 0, header=False)

        # initialize cube to hold the frames
        cube_frames = np.nan*np.ones((len(abs_sci_name_array),np.shape(test_image)[0],np.shape(test_image)[1]))
        # initialize the array to hold the parallactic angles (for de-rotation later)
        pa_array = np.nan*np.ones(len(abs_sci_name_array))
        # initialize the array to hold the frame numbers (to define masks to apply over pixel regions to make them
        #    NaNs before taking the median of a cube)
        frame_nums_array = np.ones(len(abs_sci_name_array)).astype(int)

        # loop over frames
        for frame_num in range(0,len(abs_sci_name_array)):
            print("---------------")
            print("Adding relative frame num " + str(frame_num) + " out of " + str(len(abs_sci_name_array)))
            print("Corresponding to file base name " + str(os.path.basename(abs_sci_name_array[frame_num])))

            # read in the cutout science frames
            sci, header_sci = fits.getdata(abs_sci_name_array[frame_num], 0, header=True)

            # define the mask of this science frame
            ## ## fine-tune this step later!
            mask_weird = np.ones(np.shape(sci))
            no_mask = np.copy(mask_weird) # a non-mask for reconstructing sat PSFs
            #mask_weird[sci > 55000] = np.nan # mask saturating region (~55000 for empirical PSFs)
            mask_weird[sci > 4.4e9] = np.nan ## this value just for fake data

            # check if PCA can be done at all; if not, skip this science frame
            # (N.b. We don't need a PCA reconstruction quite yet, but this is just a check.)
            # (N.b. Note this is the only criterion right now for deciding if a science
            #       frame will be used.)
            fit_2_star = fit_pca_star(self.pca_star_basis_cube, sci, mask_weird, n_PCA=1)
            if not fit_2_star:
                print("Incompatible dimensions; skipping this frame...")
                continue

            # add image to cube, add PA to array, and add frame number to array
            cube_frames[frame_num] = sci
            pa_array[frame_num] = header_sci["LBT_PARA"]
            frame_nums_array[frame_num] = int(os.path.basename(abs_sci_name_array[frame_num]).split("_")[-1].split(".")[0])

            print("type:")
            print(type(sci))


        # if writing to disk for checking
        if self.write:

            hdr = fits.Header()
            # parameters of fake planets are meaningless, since none are injected,
            # but we need these values to be populated for downstream
            hdr["ANGEOFN"] = self.fake_params["angle_deg_EofN"]
            hdr["RADASEC"] = self.fake_params["rad_asec"]
            hdr["AMPLIN"] = self.fake_params["ampl_linear_norm"]
            
            file_name = self.config_data["data_dirs"]["DIR_OTHER_FITS"] + str(saved_cube_basename)
            fits.writeto(filename = file_name,
                         data = cube_frames,
                         header = hdr,
                         overwrite = True)
            print("Wrote cube (without fake planets) to disk as " + file_name)
        
        print("Array of PA")
        print(pa_array)

        # return cube of frames and array of PAs
        return cube_frames, pa_array, frame_nums_array


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
                 abs_host_star_PCA_name,
                 abs_fake_planet_PCA_name,
                 config_data = config,
                 write = False):
        '''
        INPUTS:
        fake_params: parameters of the fake companion
        n_PCA: number of principal components to use
        abs_host_star_PCA_name: absolute file name of the PCA cube to reconstruct the host star
                       for host star subtraction
        abs_fake_planet_PCA_name: absolute file name of the PCA cube to reconstruct the host star
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

        # read in the PCA vector cubes for this series of frames
        self.pca_basis_cube_host_star, self.header_pca_basis_cube_host_star = fits.getdata(self.abs_host_star_PCA_name, 0, header=True)
        self.pca_basis_cube_fake_planet, self.header_pca_basis_cube_fake_planet = fits.getdata(self.abs_fake_planet_PCA_name, 0, header=True)

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
            ## COMMENTED THIS OUT SO THAT I CAN TEST FAKE DATA
            ## mask_weird[sci > 55000] = np.nan # mask saturating region
            ## THE BELOW FOR FAKE DATA
            mask_weird[sci > 4.5e9] = np.nan
            

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
            fit_host_star = fit_pca_star(self.pca_basis_cube_host_star, sci, no_mask, n_PCA=100)
            fit_fake_planet = fit_pca_star(self.pca_basis_cube_fake_planet, sci, mask_weird, n_PCA=100)
            if np.logical_or(not fit_host_star, not fit_fake_planet): # if the dimensions were incompatible, skip this science frame
                print("Incompatible dimensions; skipping this frame...")
                continue

            # get absolute amplitude of the host star (reconstructing over the saturated region)
            ampl_host_star = np.max(fit_fake_planet["recon_2d"])

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
                fit_fake_planet["recon_2d"],
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
                

        # convert to 32-bit float to save space
        cube_frames = cube_frames.astype(np.float32)

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

    injection = True: actually inject a fake PSF; False with just remove the host star and do ADI
    '''

    time_start = time.time()

    # make a list of ALL the centered cookie cutout files
    cookies_centered_06_directory = str(config["data_dirs"]["DIR_CENTERED"])
    cookies_centered_06_name_array = list(glob.glob(os.path.join(cookies_centered_06_directory, "*.fits")))

    ## lists of files are separated based on filter combination (there are four: A, B, C, D)

    # combination A: frames 4259-5608 & 5826-6301 (flux-saturated)
    cookies_A_only_centered_06_name_array = list(glob.glob(os.path.join(cookies_centered_06_directory, "*_004*.fits")))
    cookies_A_only_centered_06_name_array.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_005[012345]*.fits")))
    cookies_A_only_centered_06_name_array.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_0058[3456789]*.fits")))
    cookies_A_only_centered_06_name_array.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_0059*.fits")))
    cookies_A_only_centered_06_name_array.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_006[012]*.fits")))

    # combination B: frames 6303-6921 (flux-UNsaturated; use B as unsats for A)
    cookies_B_only_centered_06_name_array = list(glob.glob(os.path.join(cookies_centered_06_directory, "*_0063[123456789]*.fits")))
    cookies_B_only_centered_06_name_array.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_006[45678]*.fits")))
    cookies_B_only_centered_06_name_array.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_0069[01]*.fits")))

    # combination C: frames 7120-7734 (flux-UNsaturated; use C as unsats for D)
    cookies_C_only_centered_06_name_array = list(glob.glob(os.path.join(cookies_centered_06_directory, "*_0071[23456789]*.fits")))
    cookies_C_only_centered_06_name_array.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_007[23456]*.fits")))
    cookies_C_only_centered_06_name_array.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_0077[012]*.fits")))

    # combination D: frames 7927-11408 (flux-saturated)
    cookies_D_only_centered_06_name_array = list(glob.glob(os.path.join(cookies_centered_06_directory, "*_0079[3456789]*.fits")))
    cookies_D_only_centered_06_name_array.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_00[89]*.fits")))
    cookies_D_only_centered_06_name_array.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_01[01]*.fits")))


    # injecting fake PSFs?
    if (int(this_param_combo["rad_pix"]) == int(0)):
        # no fake PSF injection; just put frames into a cube (host star subtraction and ADI is done downstream)

        print("No fake planets being injected. (Input radius of fake planets is set to zero.)")

        # instantiate FakePlanetInjectorCube to put science frames into a cube, but no fakes are injected into the frames
        frames_in_cube = JustPutIntoCube(fake_params = this_param_combo,
                                         test_PCA_vector_name = str(config["data_dirs"]["DIR_PCA_CUBES_PSFS"] +
                                                                    'psf_PCA_vector_cookie_seqStart_006303_seqStop_006921.fits'),
                                                                    write = True)

        # filter combo A
        print('11A')
        '''
        cube_pre_removal_A, pas_array_A, frame_array_0_A = frames_in_cube(abs_sci_name_array = cookies_A_only_centered_06_name_array,
                                                                          saved_cube_basename = "simple_sci_frame_cube_A.fits")
        '''
        # filter combo B
        print('22A')
        cube_pre_removal_B, pas_array_B, frame_array_0_B = frames_in_cube(abs_sci_name_array = cookies_B_only_centered_06_name_array,
                                                                          saved_cube_basename = "simple_sci_frame_cube_B.fits")
        '''
        # filter combo C
        print('33A')
        cube_pre_removal_C, pas_array_C, frame_array_0_C = frames_in_cube(abs_sci_name_array = cookies_C_only_centered_06_name_array,
                                                                          saved_cube_basename = "simple_sci_frame_cube_C.fits")

        # filter combo D
        print('44A')
        cube_pre_removal_D, pas_array_D, frame_array_0_D = frames_in_cube(abs_sci_name_array = cookies_D_only_centered_06_name_array,
                                                                          saved_cube_basename = "simple_sci_frame_cube_D.fits")
        '''
    else:
        # inject a fake psf in each science frame, return a cube of non-derotated, non-host-star-subtracted frames

        print("-------------------------------------------------")
        print("Injecting fake planet corresponding to parameter")
        print(this_param_combo)

        # instantiate fake planet injection
        print('11B')
        cube_A_PCA_vector_name = str(config["data_dirs"]["DIR_OTHER_FITS"]
                                + "pca_cubes_psfs/"
                                + "psf_PCA_vector_cookie_seqStart_004259_seqStop_005608.fits")
        cube_B_PCA_vector_name = str(config["data_dirs"]["DIR_OTHER_FITS"]
                                + "pca_cubes_psfs/"
                                + "psf_PCA_vector_cookie_seqStart_006303_seqStop_006921.fits")
        cube_C_PCA_vector_name = str(config["data_dirs"]["DIR_OTHER_FITS"]
                                + "pca_cubes_psfs/"
                                + "psf_PCA_vector_cookie_seqStart_007120_seqStop_007734.fits")
        cube_D_PCA_vector_name = str(config["data_dirs"]["DIR_OTHER_FITS"]
                                + "pca_cubes_psfs/"
                                + "psf_PCA_vector_cookie_seqStart_007927_seqStop_011408.fits")

        inject_fake_psfs_A = FakePlanetInjectorCube(fake_params = this_param_combo,
                                          n_PCA = 100,
                                          abs_host_star_PCA_name = cube_A_PCA_vector_name,
                                          abs_fake_planet_PCA_name = cube_B_PCA_vector_name,
                                          write = False)

        # call fake planet injection
        print('22B')
        cube_pre_removal_A, pas_array_A, frame_array_0_A = inject_fake_psfs(cookies_A_only_centered_06_name_array)

        # do the same for cubes B,C,D
        print('33B')
        inject_fake_psfs_B = FakePlanetInjectorCube(fake_params = this_param_combo,
                                          n_PCA = 100,
                                          abs_host_star_PCA_name = config["data_dirs"]["DIR_OTHER_FITS"] \
                                          + "pca_cubes_psfs/" \
                                          + "psf_PCA_vector_cookie_seqStart_004259_seqStop_005600.fits",
                                          abs_fake_planet_PCA_name = [...],
                                          write = False)
        print('44B')
        cube_pre_removal_B, pas_array_B, frame_array_0_B = inject_fake_psfs(cookies_B_only_centered_06_name_array)
        print('55B')
        inject_fake_psfs_C = FakePlanetInjectorCube(fake_params = this_param_combo,
                                          n_PCA = 100,
                                          abs_host_star_PCA_name = config["data_dirs"]["DIR_OTHER_FITS"] \
                                          + "pca_cubes_psfs/" \
                                          + "psf_PCA_vector_cookie_seqStart_004259_seqStop_005600.fits",
                                          abs_fake_planet_PCA_name = [...],
                                          write = False)
        print('66B')
        cube_pre_removal_C, pas_array_C, frame_array_0_C = inject_fake_psfs(cookies_C_only_centered_06_name_array)
        print('77B')
        inject_fake_psfs_D = FakePlanetInjectorCube(fake_params = this_param_combo,
                                          n_PCA = 100,
                                          abs_host_star_PCA_name = config["data_dirs"]["DIR_OTHER_FITS"] \
                                          + "pca_cubes_psfs/" \
                                          + "psf_PCA_vector_cookie_seqStart_004259_seqStop_005600.fits",
                                          abs_fake_planet_PCA_name = [...],
                                          write = False)
        print('88B')
        cube_pre_removal_D, pas_array_D, frame_array_0_D = inject_fake_psfs(cookies_D_only_centered_06_name_array) 



    # instantiate removal of host star from each frame in the cube, whether or not
    # these are frames with fake planets
    print('55')
    ## ## START HERE

    # N.b. cube B is used as unsats for cube A
    '''
    remove_hosts_A = host_removal.HostRemovalCube(fake_params = this_param_combo,
                                                    cube_frames = cube_pre_removal_A,
                                                    n_PCA = 100,
                                                    outdir = config["data_dirs"]["DIR_FAKE_PSFS_HOST_REMOVED"],
                                                    abs_host_star_PCA_name = config["data_dirs"]["DIR_OTHER_FITS"] \
                                                          + "pca_cubes_psfs/" \
                                                          + "psf_PCA_vector_cookie_seqStart_004259_seqStop_005608.fits",
                                                    abs_fake_planet_PCA_name = config["data_dirs"]["DIR_OTHER_FITS"] \
                                                          + "pca_cubes_psfs/" \
                                                          + "psf_PCA_vector_cookie_seqStart_006303_seqStop_006921.fits",
                                                    frame_array = frame_array_0_A,
                                                    write = True)
    # call and return cube of host-removed frames
    print('66')
    removed_hosts_cube_A, frame_array_1_A = remove_hosts_A()
    '''

    # do the same for cubes B,C,D
    print('77')
    # N.b. here I just treat frames for this sequence (cube B) as both the sats and unsats for itself
    remove_hosts_B = host_removal.HostRemovalCube(fake_params = this_param_combo,
                                                    cube_frames = cube_pre_removal_B,
                                                    n_PCA = 100,
                                                    outdir = config["data_dirs"]["DIR_FAKE_PSFS_HOST_REMOVED"],
                                                    abs_host_star_PCA_name = config["data_dirs"]["DIR_OTHER_FITS"] \
                                                          + "pca_cubes_psfs/" \
                                                          + "psf_PCA_vector_cookie_seqStart_006303_seqStop_006921.fits",
                                                    abs_fake_planet_PCA_name = config["data_dirs"]["DIR_OTHER_FITS"] \
                                                          + "pca_cubes_psfs/" \
                                                          + "psf_PCA_vector_cookie_seqStart_006303_seqStop_006921.fits",
                                                    frame_array = frame_array_0_B,
                                                    write = True)
    print('88')
    removed_hosts_cube_B, frame_array_1_B = remove_hosts_B()
    print('99')
    # N.b. here I just treat frames for this sequence (cube C) as both the sats and unsats for itself
    remove_hosts_C = host_removal.HostRemovalCube(fake_params = this_param_combo,
                                                    cube_frames = cube_pre_removal_C,
                                                    n_PCA = 100,
                                                    outdir = config["data_dirs"]["DIR_FAKE_PSFS_HOST_REMOVED"],
                                                    abs_host_star_PCA_name = config["data_dirs"]["DIR_OTHER_FITS"] \
                                                          + "pca_cubes_psfs/" \
                                                          + "psf_PCA_vector_cookie_seqStart_007120_seqStop_007734.fits",
                                                    abs_fake_planet_PCA_name = config["data_dirs"]["DIR_OTHER_FITS"] \
                                                          + "pca_cubes_psfs/" \
                                                          + "psf_PCA_vector_cookie_seqStart_007120_seqStop_007734.fits",
                                                    frame_array = frame_array_0_C,
                                                    write = True)
    print('111')
    removed_hosts_cube_C, frame_array_1_C = remove_hosts_C()
    print('222')
    remove_hosts_D = host_removal.HostRemovalCube(fake_params = this_param_combo,
                                                    cube_frames = cube_pre_removal_D,
                                                    n_PCA = 100,
                                                    outdir = config["data_dirs"]["DIR_FAKE_PSFS_HOST_REMOVED"],
                                                    abs_host_star_PCA_name = config["data_dirs"]["DIR_OTHER_FITS"] \
                                                          + "pca_cubes_psfs/" \
                                                          + "psf_PCA_vector_cookie_seqStart_007927_seqStop_011408.fits",
                                                    abs_fake_planet_PCA_name = config["data_dirs"]["DIR_OTHER_FITS"] \
                                                          + "pca_cubes_psfs/" \
                                                          + "psf_PCA_vector_cookie_seqStart_007120_seqStop_007734.fits",
                                                    frame_array = frame_array_0_D,
                                                    write = True)
    print('333')
    removed_hosts_cube_D, frame_array_1_D = remove_hosts_D()

    # instantiate derotation, ADI, sensitivity determination
    print('444')
    median_instance_A = detection.MedianCube(fake_params = this_param_combo,
                                               host_subt_cube = removed_hosts_cube_A,
                                               pa_array = pas_array_A,
                                               frame_array = frame_array_1_A,
                                               write_cube = True)

    # call derotation, ADI, sensitivity determination
    print('555')
    make_median_A = median_instance_A(apply_mask_after_derot = True, fake_planet = True)
    del removed_hosts_cube_A # clear memory
    # do the same for cubes B, C, D
    print('666')
    median_instance_B = detection.MedianCube(fake_params = this_param_combo,
                                               host_subt_cube = removed_hosts_cube_B,
                                               pa_array = pas_array_B,
                                               frame_array = frame_array_1_B,
                                               write_cube = True)
    print('777')
    make_median_B = median_instance_B(apply_mask_after_derot = True, fake_planet = True)
    del removed_hosts_cube_B # clear memory
    print('888')
    median_instance_C = detection.MedianCube(fake_params = this_param_combo,
                                               host_subt_cube = removed_hosts_cube_C,
                                               pa_array = pas_array_C,
                                               frame_array = frame_array_1_C,
                                               write_cube = True)
    print('999')
    make_median_C = median_instance_C(apply_mask_after_derot = True, fake_planet = True)
    del removed_hosts_cube_C # clear memory
    print('1111')
    median_instance_D = detection.MedianCube(fake_params = this_param_combo,
                                               host_subt_cube = removed_hosts_cube_D,
                                               pa_array = pas_array_D,
                                               frame_array = frame_array_1_D,
                                               write_cube = True)
    print('2222')
    make_median_D = median_instance_D(apply_mask_after_derot = True, fake_planet = True)
    del removed_hosts_cube_D # clear memory
    
    elapsed_time = np.subtract(time.time(), time_start)

    print("----------------------------------------------------------------")
    print("Completed one fake planet parameter configuration")
    print("Elapsed time (sec): ")
    print(str(int(elapsed_time)))


#######################

def synthetic_fizeau_inject_remove_adi(this_param_combo):
    '''
    To parallelize a serial operation across cores, I need to define this function that goes through
    the fake planet injection, host star removal, and ADI steps for a given combination of fake planet parameters

    injection = True: actually inject a fake PSF; False with just remove the host star and do ADI
    '''

    time_start = time.time()

    # make a list of ALL the centered cookie cutout files
    cookies_centered_06_directory = str(config["data_dirs"]["DIR_CENTERED"])
    cookies_centered_06_name_array = list(glob.glob(os.path.join(cookies_centered_06_directory, "*.fits")))


    # injecting fake PSFs?
    if (int(this_param_combo["rad_pix"]) == int(0)):
        # no fake PSF injection; just put frames into a cube (host star subtraction and ADI is done downstream)

        print("No fake planets being injected. (Input radius of fake planets is set to zero.)")

        # instantiate FakePlanetInjectorCube to put science frames into a cube, but no fakes are injected into the frames
        frames_in_cube = JustPutIntoCube(fake_params = this_param_combo,
                                         test_PCA_vector_name = str(config["data_dirs"]["DIR_PCA_CUBES_PSFS"] +
                                                                    'psf_PCA_vector_cookie_seqStart_000000_seqStop_010000.fits'),
                                                                    write = True)

        cube_pre_removal_A, pas_array_A, frame_array_0_A = frames_in_cube(abs_sci_name_array = cookies_centered_06_name_array,
                                                                          saved_cube_basename = "simple_synthetic_sci_frame_cube_A.fits")

    else:
        # inject a fake psf in each science frame, return a cube of non-derotated, non-host-star-subtracted frames

        print("-------------------------------------------------")
        print("Injecting fake planet corresponding to parameter")
        print(this_param_combo)

        # instantiate fake planet injection
        print('11B')
        cube_A_PCA_vector_name = str(config["data_dirs"]["DIR_OTHER_FITS"]
                                + "pca_cubes_psfs/"
                                + "psf_PCA_vector_cookie_seqStart_007000_seqStop_007500.fits")

        inject_fake_psfs_A = FakePlanetInjectorCube(fake_params = this_param_combo,
                                          n_PCA = 100,
                                          abs_host_star_PCA_name = cube_A_PCA_vector_name,
                                          abs_fake_planet_PCA_name = cube_B_PCA_vector_name,
                                          write = False)

        # call fake planet injection
        print('22B')
        cube_pre_removal_A, pas_array_A, frame_array_0_A = inject_fake_psfs(cookies_A_only_centered_06_name_array)



    # instantiate removal of host star from each frame in the cube, whether or not
    # these are frames with fake planets
    ## BEGIN TEST
    import ipdb; ipdb.set_trace()
    file_name = "cube_pre_removal.fits"
    fits.writeto(filename = file_name,data = cube_pre_removal_A,overwrite = True)
    ipdb.set_trace()
    ## END TEST
            print("Wrote fake-planet-injected cube to disk as " + file_name)
    remove_hosts_A = host_removal.HostRemovalCube(fake_params = this_param_combo,
                                                    cube_frames = cube_pre_removal_A,
                                                    n_PCA = 100,
                                                    outdir = config["data_dirs"]["DIR_FAKE_PSFS_HOST_REMOVED"],
                                                    abs_host_star_PCA_name = config["data_dirs"]["DIR_PCA_CUBES_PSFS"] \
                                                          + "psf_PCA_vector_cookie_seqStart_000000_seqStop_010000.fits",
                                                    abs_fake_planet_PCA_name = config["data_dirs"]["DIR_PCA_CUBES_PSFS"] \
                                                          + "psf_PCA_vector_cookie_seqStart_000000_seqStop_010000.fits",
                                                    frame_array = frame_array_0_A,
                                                    write = True)

    removed_hosts_cube_A, frame_array_1_A = remove_hosts_A()


    # instantiate derotation, ADI, sensitivity determination
    print('444')
    median_instance_A = detection.MedianCube(fake_params = this_param_combo,
                                               host_subt_cube = removed_hosts_cube_A,
                                               pa_array = pas_array_A,
                                               frame_array = frame_array_1_A,
                                               write_cube = True)

    # call derotation, ADI, sensitivity determination
    print('555')
    make_median_A = median_instance_A(apply_mask_after_derot = True, fake_planet = True)
    del removed_hosts_cube_A # clear memory
    
    elapsed_time = np.subtract(time.time(), time_start)

    print("----------------------------------------------------------------")
    print("Completed one fake planet parameter configuration")
    print("Elapsed time (sec): ")
    print(str(int(elapsed_time)))

#######################


def main():
    '''
    Make grid of fake planet parameters, and inject fake planets

    no_injection = False: fake planets will not be injected; True means that
        ADI will  
    '''

    # configuration data
    config = configparser.ConfigParser() # for parsing values in .init file
    config.read("modules/config.ini")

    # fake planet injection parameters
    '''
    for NO injection of fake planets (and only host star removal and ADI), set rad_asec equal to zero and the others to one element each
    i.e, {"angle_deg_EofN": [0.], "rad_asec": [0.], "ampl_linear_norm": [0.]}
    '''
    fake_params_pre_permute = {"angle_deg_EofN": [0.], "rad_asec": [0.], "ampl_linear_norm": [0.]}
    '''
    fake_params_pre_permute = {"angle_deg_EofN": [270.],
                               "rad_asec": [0.5, 1.5],
                               "ampl_linear_norm": [1e-3, 1e-4, 1e-5]}
    '''

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

    # TRYING IT OVER 8 CORES AS OPPOSED TO 16 TO SEE IF I AVOID TOO MUCH MEMORY LEAKAGE
    #pool = multiprocessing.Pool(8)

    # create list of dictionaries of fake planet parameters
    # (one dictionary is fed to each core at a time)
    param_dict_list = []
    for k in range(0,len(experiment_vector)):
        param_dict_list.append(experiment_vector.iloc[k].to_dict())


    ## BEGIN THIS LINE IS A TEST ONLY
    #inject_remove_adi(param_dict_list[0])
    ## END TEST
        
    pool.map(synthetic_fizeau_inject_remove_adi, param_dict_list)



