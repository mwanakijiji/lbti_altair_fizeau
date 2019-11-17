import multiprocessing
import configparser
import glob
import time
import itertools
import pandas as pd
import pickle
import math
import datetime
import os
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
        import ipdb; ipdb.set_trace()
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
            print("injection_ADI: Adding relative frame num " + str(frame_num) + " out of " + str(len(abs_sci_name_array)))
            print("injection_ADI: Corresponding to file base name " + str(os.path.basename(abs_sci_name_array[frame_num])))

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
            fit_2_star = fit_pca_star(pca_cube = self.pca_star_basis_cube,
                                      sciImg = sci,
                                      raw_pca_training_median = np.zeros(np.shape(sci)),
                                      mask_weird = mask_weird,
                                      n_PCA=1)
            if not fit_2_star:
                print("injection_ADI: Incompatible dimensions; skipping this frame...")
                continue

            # add image to cube, add PA to array, and add frame number to array
            cube_frames[frame_num] = sci
            pa_array[frame_num] = header_sci["LBT_PARA"]
            frame_nums_array[frame_num] = int(os.path.basename(abs_sci_name_array[frame_num]).split("_")[-1].split(".")[0])

        # write median to disk for reading it in downstream, smoothing it, and finding host star amplitude
        '''
        # REMOVE THIS SECTION; IT LOOKS LIKE ITS WRITING OUT A MEDIAN WITHOUT DEROTATION
        median_just_put_into_cube = np.median(cube_frames, axis=0)
        file_name = self.config_data["data_dirs"]["DIR_OTHER_FITS"] + self.config_data["file_names"]["MEDIAN_SCI_FRAME"]
        fits.writeto(filename = file_name,
                         data = median_just_put_into_cube,
                         overwrite = True)
        print("injection_ADI: Wrote median of science frames (without fake planets or any other modification) to disk as \n" + file_name)
        print("-"*prog_bar_width)
        '''

        # if writing to disk for checking
        if self.write:

            hdr = fits.Header()
            # parameters of fake planets are meaningless, since none are injected,
            # but we need these values to be populated for downstream
            hdr["ANGEOFN"] = self.fake_params["angle_deg_EofN"]
            hdr["RADASEC"] = self.fake_params["rad_asec"]
            hdr["AMPLIN"] = self.fake_params["ampl_linear_norm"]

            file_name = str(saved_cube_basename)
            fits.writeto(filename = file_name,
                         data = cube_frames,
                         header = hdr,
                         overwrite = True)
            print("injection_ADI: "+str(datetime.datetime.now())+": Wrote cube of science frames (without fake planets or any other modification) to disk as \n"
                  + file_name)
            print("-"*prog_bar_width)

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
                injection_iteration,
                 fake_params,
                 n_PCA,
                 write_name_abs_host_star_PCA,
                 read_name_abs_fake_planet_PCA,
                 read_name_raw_pca_median,
                 config_data = config,
                 write = False):
        '''
        INPUTS:
        injection_iteration: iteration number of fake planet injection
            (0: initial injection; >=1: successive injections)
        fake_params: parameters of the fake companion
        n_PCA: number of principal components to use
        write_name_abs_host_star_PCA: absolute file name of the PCA cube to reconstruct the host star
            for host star subtraction
        read_name_abs_fake_planet_PCA: absolute file name of the PCA cube to reconstruct the host star
            for making a fake planet (i.e., without saturation effects)
        read_name_raw_pca_median: absolute file name of the raw PCA training set median (i.e., the
            offset not preserved by the PCA decomposition)
        config_data: configuration data, as usual
        write: flag as to whether data product should be written to disk (for checking)
        '''

        self.injection_iteration = injection_iteration
        self.n_PCA = n_PCA
        self.abs_host_star_PCA_name = write_name_abs_host_star_PCA
        self.abs_fake_planet_PCA_name = read_name_abs_fake_planet_PCA
        self.read_name_raw_pca_median = read_name_raw_pca_median
        self.config_data = config_data
        self.write = write

        # read in the PCA vector cubes for this series of frames
        self.pca_basis_cube_host_star, self.header_pca_basis_cube_host_star = fits.getdata(self.abs_host_star_PCA_name, 0, header=True)
        self.pca_basis_cube_fake_planet, self.header_pca_basis_cube_fake_planet = fits.getdata(self.abs_fake_planet_PCA_name, 0, header=True)

        # read in the raw PCA training set median, to
        # 1. subtract from a given PSF to leave only residuals
        # 2. add back in to the PCA-recontructed residuals and reconstruct the PSF
        self.raw_pca_basis_median = fits.getdata(self.read_name_raw_pca_median, 0, header=False)

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

        # string for making subdirectories to place ADI frames in
        if self.injection_iteration:
            injection_iteration_string = "inj_iter_" + str(self.injection_iteration).zfill(4)
        else:
            injection_iteration_string = "no_fake_planet"
        print("injection_ADI: at __init__, read in PCA vector for host star \n" +
              self.abs_host_star_PCA_name)
        print("injection_ADI: at __init__, read in PCA vector for fake planet \n" +
              self.abs_fake_planet_PCA_name)
        print("-"*prog_bar_width)

        # read in one frame to get the shape
        test_image = fits.getdata(abs_sci_name_array[0], 0, header=False)

        # initialize cube to hold the frames
        print("injection_ADI: Memory error, 0 " + str(len(abs_sci_name_array)))
        print("injection_ADI: Memory error, shape " + str(np.shape(test_image)))
        cube_frames = np.nan*np.ones((len(abs_sci_name_array),np.shape(test_image)[0],np.shape(test_image)[1]))
        # initialize the array to hold the parallactic angles (for de-rotation later)
        pa_array = np.nan*np.ones(len(abs_sci_name_array))
        # initialize the array to hold the frame numbers (to define masks to apply over pixel regions to make them
        #    NaNs before taking the median of a cube)
        frame_nums_array = np.ones(len(abs_sci_name_array)).astype(int)

        # loop over frames to inject fake planets in each of them
        for frame_num in range(0,len(abs_sci_name_array)):
            print("injection_ADI: Injecting a fake planet into cube slice " + str(frame_num))
            print(" which corresponds to file \n" + abs_sci_name_array[frame_num])

            # read in the cutout science frames
            sci, header_sci = fits.getdata(abs_sci_name_array[frame_num], 0, header=True)
            #
            # define the mask of this science frame
            ## ## fine-tune this step later!
            mask_weird = np.ones(np.shape(sci))
            no_mask = np.copy(mask_weird) # a non-mask for reconstructing sat PSFs
            ## COMMENTED THIS OUT SO THAT I CAN TEST FAKE DATA
            ## mask_weird[sci > 55000] = np.nan # mask saturating region
            ## THE BELOW FOR FAKE DATA
            mask_weird[sci > 4.5e9] = np.nan
            #

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
            print("injection_ADI: Applying this PCA basis for the host star: \n" +
                  str(self.abs_host_star_PCA_name))
            fit_host_star = fit_pca_star(pca_cube = self.pca_basis_cube_host_star,
                                         sciImg = sci,
                                         raw_pca_training_median = self.raw_pca_basis_median,
                                         mask_weird = no_mask,
                                         n_PCA=100)
            print("injection_ADI: Applying this PCA basis for the fake planet: \n" +
                  str(self.abs_fake_planet_PCA_name))
            print("-"*prog_bar_width)
            fit_fake_planet = fit_pca_star(pca_cube = self.pca_basis_cube_fake_planet,
                                           sciImg = sci,
                                           raw_pca_training_median = self.raw_pca_basis_median,
                                           mask_weird = mask_weird,
                                           n_PCA=100)
            if np.logical_or(not fit_host_star, not fit_fake_planet): # if the dimensions were incompatible, skip this science frame
                print("injection_ADI: Incompatible dimensions; skipping this frame...")
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

            #print('pa in header:')
            #print(header_sci["LBT_PARA"])
            #print('fake angle E of N:')
            #print(fake_angle_e_of_n_deg)

            # shift the PSF image to the location of the fake planet
            reconImg_shifted = scipy.ndimage.interpolation.shift(
                fit_fake_planet["recon_2d"],
                shift = [self.fake_params["y_pix_coord"],
                         self.fake_params["x_pix_coord"]]) # shift in +y,+x convention

            #print('fake_params y x')
            #print(self.fake_params["y_pix_coord"])
            #print(self.fake_params["x_pix_coord"])

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

            # check if injection_iteration_string exists; if not, make the directory
            abs_path_name = self.config_data["data_dirs"]["DIR_OTHER_FITS"] + \
                            injection_iteration_string + "/"
            if not os.path.exists(abs_path_name):
                os.makedirs(abs_path_name)
                print("Made directory " + abs_path_name)
            file_name = self.config_data["data_dirs"]["DIR_OTHER_FITS"] + \
                injection_iteration_string + "/" + \
                "fake_planet_injected_cube_" + \
                str(self.fake_params["angle_deg_EofN"]) + "_" + \
                str(self.fake_params["rad_asec"]) + \
                "_" + str(self.fake_params["ampl_linear_norm"]) + ".fits"
            fits.writeto(filename = file_name,
                         data = cube_frames,
                         header = hdr,
                         overwrite = True)
            print("injection_ADI: "+str(datetime.datetime.now())+"Wrote fake-planet-injected cube to disk as " + file_name)
            print("-"*prog_bar_width)

        #print("injection_ADI: Array of PA")
        #print(pa_array)

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
    if (int(this_param_combo["rad_asec"]) == int(0)):
        # no fake PSF injection; just put frames into a cube (host star subtraction and ADI is done downstream)

        print("injection_ADI: No fake planets being injected. (Input radius of fake planets is set to zero.)")

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
        print("injection_ADI: Injecting fake planet corresponding to parameter")
        print(this_param_combo)

        # instantiate fake planet injection
        print('11B')
        write_name_abs_cube_A_PCA_vector = str(config["data_dirs"]["DIR_OTHER_FITS"]
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
                                          abs_host_star_PCA_name = write_name_abs_cube_A_PCA_vector,
                                          abs_fake_planet_PCA_name = cube_B_PCA_vector_name,
                                          write = False)

        # call fake planet injection
        print('22B')
        cube_pre_removal_A, pas_array_A, frame_array_0_A = inject_fake_psfs_A(cookies_A_only_centered_06_name_array)

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
        cube_pre_removal_B, pas_array_B, frame_array_0_B = inject_fake_psfs_B(cookies_B_only_centered_06_name_array)
        print('55B')
        inject_fake_psfs_C = FakePlanetInjectorCube(fake_params = this_param_combo,
                                          n_PCA = 100,
                                          abs_host_star_PCA_name = config["data_dirs"]["DIR_OTHER_FITS"] \
                                          + "pca_cubes_psfs/" \
                                          + "psf_PCA_vector_cookie_seqStart_004259_seqStop_005600.fits",
                                          abs_fake_planet_PCA_name = [...],
                                          write = False)
        print('66B')
        cube_pre_removal_C, pas_array_C, frame_array_0_C = inject_fake_psfs_C(cookies_C_only_centered_06_name_array)
        print('77B')
        inject_fake_psfs_D = FakePlanetInjectorCube(fake_params = this_param_combo,
                                          n_PCA = 100,
                                          abs_host_star_PCA_name = config["data_dirs"]["DIR_OTHER_FITS"] \
                                          + "pca_cubes_psfs/" \
                                          + "psf_PCA_vector_cookie_seqStart_004259_seqStop_005600.fits",
                                          abs_fake_planet_PCA_name = [...],
                                          write = False)
        print('88B')
        cube_pre_removal_D, pas_array_D, frame_array_0_D = inject_fake_psfs_D(cookies_D_only_centered_06_name_array)



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
    print("injection_ADI: Completed one fake planet parameter configuration")
    print("injection_ADI: Elapsed time (sec): ")
    print(str(int(elapsed_time)))


#######################

class SyntheticFizeauInjectRemoveADI:

    def __init__(self,
                injection_iteration,
                 file_name_list,
                 n_PCA_host_removal,
                 read_name_abs_test_PCA_vector,
                 write_name_abs_cube_put_frames_into_it_simple,
                 write_name_abs_cube_A_PCA_vector,
                 read_name_abs_pca_pre_decomposition_median,
                 write_name_abs_derotated_sci_median,
                 write_name_abs_host_star_PCA,
                 read_name_abs_fake_planet_PCA,
                 read_name_abs_pca_tesselation_pattern):
        '''
        injection_iteration: number of the fake planet injection iteration
            (None if no planets are being injected)
        file_name_list: list of names of files to operate on
        n_PCA_host_removal: number of PCA modes to use for subtracting out the host star
        read_name_abs_test_PCA_vector: name of a test PCA vector cube file just to see if
            decomposition can be done at all
        write_name_abs_cube_put_frames_into_it_simple: if no fakes are being injected,
            this file name contains the stack of all the frames
        write_name_abs_cube_A_PCA_vector: name of PCA vector
        read_name_abs_pca_pre_decomposition_median: median of non-derotated science frames, to
            subtract from frames before PCA decomposition
        write_name_abs_derotated_sci_median: name of median of derotated science frames, to find
            host star amplitude and to reconstruct PSFs (since PCA basis set only reconstructs
            residuals)
        write_name_abs_host_star_PCA: name of PCA cube to use for host star decomposition, for
            subtraction
        read_name_abs_fake_planet_PCA: name of PCA cube to use for host star decomposition
            such that the full PSF is reconstructed (like to make fake planets)
        read_name_abs_pca_tesselation_pattern: name of tesselation cube
        '''

        self.injection_iteration = injection_iteration
        self.cookies_centered_06_name_array = file_name_list
        self.n_PCA_host_removal = n_PCA_host_removal
        self.test_PCA_vector_name = read_name_abs_test_PCA_vector
        self.cube_put_frames_into_it_simple_name = write_name_abs_cube_put_frames_into_it_simple
        self.write_name_abs_cube_A_PCA_vector = write_name_abs_cube_A_PCA_vector
        self.pca_pre_decomposition_median_name = read_name_abs_pca_pre_decomposition_median
        self.write_name_abs_derotated_sci_median = write_name_abs_derotated_sci_median
        self.abs_host_star_PCA_name = write_name_abs_host_star_PCA
        self.read_name_abs_fake_planet_PCA = read_name_abs_fake_planet_PCA
        self.read_name_abs_pca_tesselation_pattern = read_name_abs_pca_tesselation_pattern

    def __call__(self, this_param_combo):
        '''
        To parallelize a serial operation across cores, I need to define this class
        that goes through the fake planet injection, host star removal, and ADI steps
        for a given combination of fake planet parameters

        injection = True: actually inject a fake PSF; False with just remove the host star
            and do ADI
        '''

        time_start = time.time()

        #import ipdb; ipdb.set_trace()
        # injecting fake PSFs?
        if (int(this_param_combo["rad_asec"]) == int(0)):
            # no fake PSF injection; just put frames into a cube (host star subtraction and ADI is done downstream)
            print("injection_ADI: No fake planets being injected. (Input radius of fake planets is set to zero.)")

            # instantiate FakePlanetInjectorCube to put science frames into a cube, but no fakes are injected into the frames
            frames_in_cube = JustPutIntoCube(fake_params = this_param_combo,
                                             test_PCA_vector_name = self.test_PCA_vector_name,
                                             write = True)
            import ipdb; ipdb.set_trace()
            cube_pre_removal_A, pas_array_A, frame_array_0_A = frames_in_cube(abs_sci_name_array = self.cookies_centered_06_name_array,
                                                                              saved_cube_basename = self.cube_put_frames_into_it_simple_name)

        else:
            # inject a fake psf in each science frame, return a cube of non-derotated, non-host-star-subtracted frames
            print("injection_ADI: Injecting fake planet corresponding to parameter")
            print(this_param_combo)

            # instantiate fake planet injection
            inject_fake_psfs_A = FakePlanetInjectorCube(
                                        injection_iteration = self.injection_iteration,
                                        fake_params = this_param_combo,
                                          n_PCA = 100,
                                          write_name_abs_host_star_PCA = self.write_name_abs_cube_A_PCA_vector,
                                          read_name_abs_fake_planet_PCA = self.write_name_abs_cube_A_PCA_vector,
                                          read_name_raw_pca_median = self.pca_pre_decomposition_median_name,
                                          write = False)

            # call fake planet injection
            cube_pre_removal_A, pas_array_A, frame_array_0_A = inject_fake_psfs_A(self.cookies_centered_06_name_array)

        # instantiate removal of host star from each frame in the cube, whether or not
        # these are frames with fake planets

        ## BEGIN TEST
        #import ipdb; ipdb.set_trace()
        #file_name = "junk_cube_pre_removal.fits"
        #fits.writeto(filename = file_name,data = cube_pre_removal_A,overwrite = True)
        #print("Wrote cube pre removal")
        #print(cookies_centered_06_name_array)
        #ipdb.set_trace()
        ## END TEST
        ####### START HERE
        # subtract the PCA training cube median from the cube of science frames
        # (this frame was subtracted from that cube before the PCA basis set was made)
        ## ## weak point here: this median name is hard-coded

        median_frame, header_median_frame = fits.getdata(self.pca_pre_decomposition_median_name, 0, header=True)
        print("injection_ADI: "+str(datetime.datetime.now())+": Median frame being subtracted from the cube of science frames is read in as\n" +
              self.pca_pre_decomposition_median_name)
        print("-"*prog_bar_width)
        cube_pre_removal_A_post_pca_median_removal = np.subtract(cube_pre_removal_A, median_frame)

        ###############################################################################################
        # instantiate MedianCube for derotating 'raw' science frames and taking the median (before any host
        # star was subtracted), for determining host star amplitude
        # (note this is being repeated each time a fake planet is injected; it's not efficient, but I
        # don't know of a better/clearer way of doing it)
        median_instance_sci = detection.MedianCube(injection_iteration = None,
                                                fake_params = this_param_combo,
                                               host_subt_cube = cube_pre_removal_A,
                                               pa_array = pas_array_A,
                                               frame_array = frame_array_0_A,
                                               write_cube = True)
        print("injection_ADI: "+str(datetime.datetime.now())
              +": Writing out median of derotated 'raw' science frames, for finding host star amplitude, as\n"
              +self.write_name_abs_derotated_sci_median)
        make_median_sci = median_instance_sci(adi_write_name = self.write_name_abs_derotated_sci_median,
                                          apply_mask_after_derot = True,
                                          fake_planet = True)
        print("-"*prog_bar_width)

        ###############################################################################################
        # now actually do the host star subtraction
        remove_hosts_A = host_removal.HostRemovalCube(injection_iteration = self.injection_iteration,
                                                    fake_params = this_param_combo,
                                                    cube_frames = cube_pre_removal_A_post_pca_median_removal,
                                                    n_PCA = self.n_PCA_host_removal,
                                                    outdir = config["data_dirs"]["DIR_FAKE_PSFS_HOST_REMOVED"],
                                                    write_name_abs_host_star_PCA = self.abs_host_star_PCA_name,
                                                    abs_fake_planet_PCA_name = self.read_name_abs_fake_planet_PCA,
                                                    abs_region_mask_name = self.read_name_abs_pca_tesselation_pattern,
                                                    frame_array = frame_array_0_A,
                                                    subtract_median_PCA_training_frame = True,
                                                    write = True)
        removed_hosts_cube_A, frame_array_1_A = remove_hosts_A()
        print("injection_ADI: Done with host removal from cube of science frames.")
        print("-"*prog_bar_width)
        # instantiate derotation, ADI, sensitivity determination of host-star-subtracted frames
        median_instance_A = detection.MedianCube(injection_iteration = self.injection_iteration,
                                                fake_params = this_param_combo,
                                               host_subt_cube = removed_hosts_cube_A,
                                               pa_array = pas_array_A,
                                               frame_array = frame_array_1_A,
                                               write_cube = True)
        # call derotation, ADI, sensitivity determination
        make_median_A = median_instance_A(apply_mask_after_derot = True,
                                      fake_planet = True)
        del removed_hosts_cube_A # clear memory

        elapsed_time = np.subtract(time.time(), time_start)

        print("injection_ADI: Completed one fake planet parameter configuration")
        print("injection_ADI: Elapsed time (sec): ")
        print(str(int(elapsed_time)))
        print("-"*prog_bar_width)



def main(inject_iteration=None):
    '''
    Make grid of fake planet parameters, and inject fake planets

    #no_injection = False: fake planets will not be injected; True means that
    #    ADI will

    INPUT:
    inject_iteration=None: no fake planets are being injected; only ADI is being performed
    inject_iteration=int: the number of the iteration for injecting fake planets; iterations
        continue until convergence to desired S/N
    '''

    # configuration data
    config = configparser.ConfigParser() # for parsing values in .init file
    config.read("modules/config.ini")

    # name of file to which we will append all S/N calculations, for each fake planet parameter
    # (not used if inject_iteration==None):
    csv_file_name = str(config["data_dirs"]["DIR_S2N"] + config["file_names"]["DETECTION_CSV"])

    if not inject_iteration:
        # if NOT injecting fake planets (and only doing host star removal and ADI), set rad_asec equal
        # to zero and the others to one element each
        fake_params_pre_permute = {"angle_deg_EofN": [0.], "rad_asec": [0.], "ampl_linear_norm": [0.]}

    if (inject_iteration == 0):
        # case of fake planet injection, first pass: inject them based on all permutations of user-given
        # parameters, permutate values of fake planet parameters to get all possible combinations

        # fake planet injection starting parameters
        fake_params_pre_permute = {"angle_deg_EofN": [0.],
                               "rad_asec": [0.30,0.35,0.40],
                               "ampl_linear_norm": [1e-3]}

        keys, values = zip(*fake_params_pre_permute.items()) # permutate
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    if (inject_iteration > 0):
        # case of fake planet injection, N>1 pass: use adjusted amplitudes, companion-by-companion, and re-inject
        # (does not make permutations of user-given parameters; this adjusts amplitudes individually)

        sn_thresh = float(config["reduc_params"]["SN_THRESHOLD"])
        #import ipdb; ipdb.set_trace()
        # csv to record ALL iteration info, but is not the initial file written
        csv_file_name_all_iters = str(config["data_dirs"]["DIR_S2N"] + \
                                config["file_names"]["DETECTION_CSV_ALL_ITER"])

        # read in detection() csv file
        noise_data = pd.read_csv(csv_file_name_all_iters, index_col=0)
        noise_data.reset_index(inplace=True,drop=True)
        # make non-redundant array of (radius,azimuth)
        ang_rad_df = noise_data.drop_duplicates(subset=["angle_deg","rad_asec"])

        #if (inject_iteration == 1):
        # initialize list of fake companion parameters
        experiments = []

        # loop over each fake companion
        for rad_az_num in range(0,len(ang_rad_df)):
            # For each combination of (radius,azimuth), iterate injected fake companion.
            # Basic algorithm:
            # 1. In the very first (N=1 step) S/N determination of fake planet, if
            #    Case 1A:  S/N is above the threshold, take the largest fake companion
            #                amplitude step (which is the same as the initial one) and
            #                subtract it to get a smaller total amplitude
            #    Case 1B:  Same as 1A, with [above->below, subtract->add, smaller->larger]
            # 2. In the N>1 step, if
            #    Case 2A:  S/N remains below/above the threshold (from the last step to this one),
            #                do the same again.
            #    Case 2B:  S/N has now crossed over the threshold compared with the last step,
            #                make step size one smaller and add/subtract it with a sign determined
            #                by the side of the S/N threshold it's on. (Ex.: If the S/N crossed from
            #                above to below the threshold level, make the next companion amplitude
            #                smaller and ADD it, to make the companion 'turn around' and seek the
            #                S/N threshold.)
            # 3. Repeat 2. to a cutoff criterion

            # retrieve the row corresponding to the most recent iteration corresponding to this (rad,az)
            old_companion_rows_all_iterations = noise_data[(noise_data["rad_asec"] == ang_rad_df.iloc[rad_az_num]["rad_asec"]) &
                                           (noise_data["angle_deg"] == ang_rad_df.iloc[rad_az_num]["angle_deg"])]
            # ('minus_1': one step back in time)
            idx_1 = np.where(old_companion_rows_all_iterations["inject_iteration"] == old_companion_rows_all_iterations["inject_iteration"].max())
            old_companion_row_minus_1 = old_companion_rows_all_iterations.iloc[idx_1]

            # assign new iteration number
            #inject_iteration = 1+np.nanmax(old_companion_rows_all_iterations["inject_iteration"])

            # initialize a new dictionary corresponding to this (rad,az)
            col_names = noise_data.columns
            new_companion_row = pd.DataFrame(np.nan, index=[0], columns=col_names)

            # Case of first iteration of fake planet amplitude
            if (inject_iteration == 1):
                this_amp_step_unsigned = np.nanmax(del_amplitude_progression)
                if (ang_rad_df.iloc[rad_az_num]["s2n"] > sn_thresh):
                    #  Case 1A: S/N > threshold -> make companion amplitude smaller by largest step
                    this_amp_step_signed = -this_amp_step_unsigned

                elif (ang_rad_df.iloc[rad_az_num]["s2n"] < sn_thresh):
                    #  Case 1B: S/N > threshold -> make companion amplitude larger by largest step
                    this_amp_step_signed = this_amp_step_unsigned

                new_companion_row["ampl_linear_norm"] = old_companion_row_minus_1["ampl_linear_norm"].values[0] + this_amp_step_signed
                new_companion_row["last_ampl_step_signed"] = this_amp_step_signed

            # Case of N>1 iteration, where comparison is made with previous step
            elif (inject_iteration > 1):
                # ('minus_2': two steps back in time)
                all_iteration_nums_sorted = np.sort(old_companion_rows_all_iterations["inject_iteration"].values)
                # get the row corresponding to this companion, two iteration steps back
                #idx_2 = np.where(old_companion_rows_all_iterations["inject_iteration"] == all_iteration_nums_sorted[-2])
                #old_companion_row_minus_2 = old_companion_rows_all_iterations.iloc[idx_2]
                #import ipdb; ipdb.set_trace()

                if (np.sign(sn_thresh - old_companion_row_minus_1["s2n"].iloc[0]) ==
                    np.sign(old_companion_row_minus_1["last_ampl_step_signed"].iloc[0])):
                    # Case 2A: S/N remained below/above the threshold, just take the same step again
                    this_amp_step_signed = old_companion_row_minus_1["last_ampl_step_signed"].iloc[0]

                elif (np.sign(sn_thresh - old_companion_row_minus_1["s2n"].iloc[0]) ==
                      -np.sign(old_companion_row_minus_1["last_ampl_step_signed"].iloc[0])):
                    # Case 2B: There is a crossover relative to the threshold S/N; make the step smaller and go the opposite way
                    # take user-defined amplitude steps and remove the previous, larger steps
                    #import ipdb; ipdb.set_trace()
                    indices_of_interest = np.where(np.array(del_amplitude_progression) < old_companion_row_minus_1["last_ampl_step_unsigned"].iloc[0])
                    # take the maximum step value left over
                    #import ipdb; ipdb.set_trace()
                    if (len(indices_of_interest[0]) == 0):
                        # if there is no more smaller amplitude change, go to
                        # next item in the loop
                        continue
                    else:
                        this_amp_step_unsigned = np.nanmax(del_amplitude_progression[indices_of_interest])
                        this_amp_step_signed = -np.sign(old_companion_row_minus_1["last_ampl_step_signed"].iloc[0])*this_amp_step_unsigned

                # add the step to get a new absolute fake companion amplitude
                new_companion_row["ampl_linear_norm"] = np.add(this_amp_step_signed,old_companion_row_minus_1["ampl_linear_norm"].iloc[0])

                # the new amplitude change will be the 'last' one after feeding the ADI frame through the detection module
                new_companion_row["last_ampl_step_signed"] = this_amp_step_signed

            #import ipdb; ipdb.set_trace()
            # keep other relevant info, regardless of iteration number
            new_companion_row["angle_deg"] = old_companion_row_minus_1["angle_deg"].values[0]
            new_companion_row["rad_asec"] = old_companion_row_minus_1["rad_asec"].values[0]
            new_companion_row["host_ampl"] = old_companion_row_minus_1["host_ampl"].values[0]
            new_companion_row["inject_iteration"] = inject_iteration

            # convert radii in asec to pixels
            ## ## functionality of polar_to_xy will need to be checked, since I changed the convention in the init
            ## ## file to be deg E of N, and in asec
            ## ## IS RAD_PIX REALLY NECESSARY HERE?
            new_companion_row["rad_pix"] = np.divide(new_companion_row["rad_asec"].values[0],np.float(config["instrum_params"]["LMIR_PS"]))
            # append new row to larger dataframe
            noise_data = noise_data.append(new_companion_row, ignore_index=True, sort=True)

            # make a dictionary of the new parameters for one companion, and
            # append it to the list of dictionaries corresponding to this
            # companion iteration
            fake_params_1_comp_dict = {"angle_deg_EofN": old_companion_row_minus_1["angle_deg"].values[0],
                                        "rad_asec": old_companion_row_minus_1["rad_asec"].values[0],
                                        "rad_pix": new_companion_row["rad_pix"].values[0],
                                        "ampl_linear_norm": new_companion_row["ampl_linear_norm"].values[0]}
            experiments.append(fake_params_1_comp_dict)
        # end loop over every fake companion, for one aplitude iteration

        # write to csv file (note it will overwrite), with NaNs which will get
        # filled in by detection module; note header
        if (inject_iteration == 1):
            #import ipdb; ipdb.set_trace()
            exists = os.path.isfile(csv_file_name_all_iters)
            if exists:
                input("A fake planet detection CSV file (for all iterated info) "+\
                    "already exists! Hit [Enter] to delete it and continue.")
                os.remove(csv_file_name_all_iters)
            noise_data.to_csv(csv_file_name_all_iters, sep = ",", mode = "w", header=True)
        elif (inject_iteration > 1):
            # write out all the data to a pre-existing csv, and don't put in new headers
            noise_data.to_csv(csv_file_name_all_iters, sep = ",", mode = "w", header=True)
        print("Wrote data on iterated companion amplitudes to csv ")
        print(str(csv_file_name_all_iters))
        print("-"*prog_bar_width)

        # N.b. the new_companion_row does not contain S/N information yet, which must be calculated by detection.py

    # remove 'none' element from initialization
    experiments = [i for i in experiments if len(i)>0]

    # convert to dataframe
    experiment_vector = pd.DataFrame(experiments)

    # clear
    del experiments
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
    # make a list of ALL the centered cookie cutout files
    import ipdb; ipdb.set_trace()
    if (inject_iteration == None):
        injection_iteration_string = "no_fake_planet"
        # the string is not being appended to the path, to avoid breakage
        # with pipeline upstream
        print("injection_ADI: No fake planet being injected")
        cookies_centered_06_directory = str(config["data_dirs"]["DIR_CENTERED"])
    elif (inject_iteration is not None):
        # if we are injecting fake planets, get source images from previous iteration
        injection_iteration_string = "inj_iter_" + str(inject_iteration).zfill(4)
        print("injection_ADI: Fake planet injection iteration number " + injection_iteration_string)
        if (inject_iteration == 0):
            # source directory is still just the original centered frames
            cookies_centered_06_directory = str(config["data_dirs"]["DIR_CENTERED"])
        elif (inject_iteration >= 1):
            # source directory is now previous iteration
            prev_iteration_string = "inj_iter_" + str(prev_iteration_string).zfill(4)
            cookies_centered_06_directory = str(config["data_dirs"]["DIR_CENTERED"]) + \
                                            prev_iteration_string

    cookies_centered_06_name_array = list(glob.glob(os.path.join(cookies_centered_06_directory, "*.fits")))

    # instantiate
    synthetic_fizeau_inject_remove_adi = SyntheticFizeauInjectRemoveADI(
        injection_iteration = inject_iteration,
        file_name_list = cookies_centered_06_name_array,
        n_PCA_host_removal = 100,
        read_name_abs_test_PCA_vector = str(config["data_dirs"]["DIR_PCA_CUBES_PSFS"] +
                                            'psf_PCA_vector_cookie_seqStart_000000_seqStop_010000.fits'),
        write_name_abs_cube_put_frames_into_it_simple = str(config["data_dirs"]["DIR_PCA_CUBES_PSFS"]
                                                            + "simple_synthetic_sci_frame_cube_A.fits"),
        write_name_abs_cube_A_PCA_vector = str(config["data_dirs"]["DIR_PCA_CUBES_PSFS"]
                                                + "psf_PCA_vector_cookie_seqStart_007000_seqStop_007500.fits"),
        read_name_abs_pca_pre_decomposition_median = str(config["data_dirs"]["DIR_PCA_CUBES_PSFS"]
                                                          + 'median_frame_seqStart_00000_seqStop_10000_pcaNum_100_host_recon.fits'),
        write_name_abs_derotated_sci_median = str(config["data_dirs"]["DIR_OTHER_FITS"]
                                                  + config["file_names"]["MEDIAN_SCI_FRAME"]),
        write_name_abs_host_star_PCA = str(config["data_dirs"]["DIR_PCA_CUBES_PSFS"]
                                           + "psf_PCA_vector_cookie_seqStart_00000_seqStop_10000_pcaNum_100_host_resids.fits"),
        read_name_abs_fake_planet_PCA = str(config["data_dirs"]["DIR_PCA_CUBES_PSFS"]
                                            + "psf_PCA_vector_cookie_seqStart_00000_seqStop_10000_pcaNum_100_host_recon.fits"),
        read_name_abs_pca_tesselation_pattern = str(config["data_dirs"]["DIR_OTHER_FITS"] +
                                                    "tesselation_10_psfs_in_each_region.fits")
        )
    '''
    Note that tesselation region options are
    mask_100x100pix_whole_frame.fits
    mask_100x100_4quad.fits
    mask_quad4_circ.fits
    mask_quad4_circ_ring.fits
    tesselation_10_psfs_in_each_region.fits
    mask_10x10_100squares.fits
    '''

    ## ## BEGIN TEST
    for param_num in range(0,len(param_dict_list)):
        print(":")
        synthetic_fizeau_inject_remove_adi(param_dict_list[param_num]) # test on just one at a time
    ## ## END TEST

    # run
    '''
    pool.map(synthetic_fizeau_inject_remove_adi, param_dict_list)
    '''
