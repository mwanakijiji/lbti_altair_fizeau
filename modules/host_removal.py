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
    PCA-decompose a saturated host star PSF and remove it, using images written to disk
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
        try:
            fit_unsat = fit_pca_star(self.pca_basis_cube_sat, sci, no_mask, n_PCA=100)
        except:
            return

        # subtract the PCA-reconstructed host star
        image_host_removed = np.subtract(sci,fit_unsat["recon_2d"])

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


class HostRemovalCube:
    '''
    PCA-decompose a saturated host star PSF and remove it, using a cube of frames in memory
    N.b. This does no de-rotation; is blind to parallactic angle
    '''

    def __init__(self,
                 fake_params,
                 cube_frames,
                 n_PCA,
                 outdir,
                 abs_host_star_PCA_name,
                 abs_fake_planet_PCA_name,
                 frame_array,
                 config_data = config,
                 write = False):
        '''
        INPUTS:
        fake_params: fake planet parameters (if applicable; this is just for making the
            file name string if we are writing out to disk; if not applicable, but
            in some other size-3 Pandas DataFrame of strings)
        cube_frames: the cube of frames to use, before host-star subtraction
        n_PCA: number of principal components to use
        outdir: directory to deposit the host-subtracted images in (this has to be
                       defined at the function call because the images may or may not
                       contain fake planet PSFs, and I want to keep them separate)
        abs_host_star_PCA_name: absolute file name of the PCA cube to reconstruct the host
                       star for host star subtraction (i.e., this is probably with saturation)
        abs_fake_planet_PCA_name: absolute file name of the PCA cube to reconstruct the host star
                       to make a fake planet (i.e., without saturation effects)
        frame_num_array: array of integers corresponding to the frame file name numbers
        config_data: configuration data, as usual
        write: flag as to whether data product should be written to disk (for checking)
        '''

        self.fake_params = fake_params
        self.cube_frames = cube_frames
        self.n_PCA = n_PCA
        self.outdir = outdir
        #self.abs_PCA_name = abs_PCA_name
        self.abs_host_star_PCA_name = abs_host_star_PCA_name
        self.abs_fake_planet_PCA_name = abs_fake_planet_PCA_name
        self.frame_num_array = frame_array
        self.config_data = config_data
        self.write = write

        # read in the PCA vector cube for this series of frames
        # (note the PCA needs to correspond to saturated PSFs, since I am subtracting
        # saturated PSFs away)
        self.pca_basis_cube_host_star, self.header_pca_basis_cube_host_star = fits.getdata(self.abs_host_star_PCA_name, 0, header=True)
        self.pca_basis_cube_fake_planet, self.header_pca_basis_fake_planet = fits.getdata(self.abs_fake_planet_PCA_name, 0, header=True)

        ##########


    def __call__(self):
        '''
        Reconstruct and subtract the host star from each slice
        
        INPUTS:
        abs_sci_name: the absolute path of the science frame into which we want to inject a planet

        OUTPUTS:
        host_subt_cube: a cube of non-derotated frames
        '''

        # make a cube that is the same shape as the input
        host_subt_cube = np.nan*np.ones(np.shape(self.cube_frames))

        # remove the host star from each slice
        for slice_num in range(0,len(self.cube_frames)):

            # select the slice from the cube
            # (there should be no masking of this frame downstream)
            sci = self.cube_frames[slice_num,:,:]

            # define the mask of this science frame
            ## ## fine-tune this step later!
            mask_weird = np.ones(np.shape(sci))
            no_mask = np.copy(mask_weird) # a non-mask for reconstructing saturated PSFs
            #mask_weird[sci > 1e8] = np.nan # mask saturating region

            ## TEST: WRITE OUT
            #hdu = fits.PrimaryHDU(sci)
            #hdulist = fits.HDUList([hdu])
            #hdu.writeto("junk_mask.fits", clobber=True)
            ## END TEST

            ###########################################
            # PCA-decompose the host star PSF
            # (note no de-rotation of the image here)

            # do the PCA fit of masked host star
            # returns dict: 'pca_vector': the PCA best-fit vector; and 'recon_2d': the 2D reconstructed PSF
            # N.b. PCA reconstruction will be to get an UN-sat PSF; note PCA basis cube involves unsat PSFs

            try:
                # fit to the host star for subtraction
                fit_host_star = fit_pca_star(self.pca_basis_cube_host_star, sci, no_mask, n_PCA=100)
            except:
                continue

            # subtract the PCA-reconstructed host star
            image_host_removed = np.subtract(sci,fit_host_star["recon_2d"])

            # pickle the PCA vector
            '''
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
            '''

            host_subt_cube[slice_num,:,:] = image_host_removed

        # if writing to disk for checking
        if self.write:

            file_name = self.config_data["data_dirs"]["DIR_OTHER_FITS"] + "host_removed_cube_" + \
              str(self.fake_params["angle_deg_EofN"]) + "_" + str(self.fake_params["rad_asec"]) + \
              "_" + str(self.fake_params["ampl_linear_norm"]) + ".fits"

            hdr = fits.Header()
            hdr["ANGEOFN"] = self.fake_params["angle_deg_EofN"]
            hdr["RADASEC"] = self.fake_params["rad_asec"]
            hdr["AMPLIN"] = self.fake_params["ampl_linear_norm"]
              
            fits.writeto(filename = file_name,
                         data = host_subt_cube,
                         header = hdr,
                         overwrite = True)
            print("Wrote host-removed-cube to disk as " + file_name)
            
        # for memory's sake
        del self.cube_frames

        print("Returning cube of host-removed frames ")

        # return
        # host_subt_cube: cube of non-derotated, host-star-subtracted frames
        # self.frame_num_array: array of the file name frame numbers (these are just passed without modification) 
        return host_subt_cube, self.frame_num_array


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

    '''
    # make a list of the images WITHOUT fake planets
    # (these are just the centered frames)
    cookies_centered_06_directory = str(config["data_dirs"]["DIR_CENTERED"])
    cookies_centered_06_name_array = list(glob.glob(os.path.join(cookies_centered_06_directory, "*.fits")))
    # separate by cube: A, B, C, or D
    sci_frames_for_cube_A = list(glob.glob(os.path.join(cookies_centered_06_directory, "*_004*.fits")))
    sci_frames_for_cube_A.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_005[012345]*.fits")))
    sci_frames_for_cube_A.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_00560[012345678]*.fits")))
    sci_frames_for_cube_A.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_00582[6789]*.fits")))
    sci_frames_for_cube_A.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_0058[3456789]*.fits")))
    sci_frames_for_cube_A.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_0059*.fits")))
    sci_frames_for_cube_A.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_006[012]*.fits")))
    sci_frames_for_cube_A.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_00630[01]*.fits")))

    sci_frames_for_cube_B = list(glob.glob(os.path.join(cookies_centered_06_directory, "*_00630[3456789]*.fits")))
    sci_frames_for_cube_B.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_006[45678]*.fits")))
    sci_frames_for_cube_B.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_0069[01]*.fits")))
    sci_frames_for_cube_B.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_00692[01]*.fits")))

    sci_frames_for_cube_C = list(glob.glob(os.path.join(cookies_centered_06_directory, "*_0071[23456789]*.fits")))
    sci_frames_for_cube_C.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_007[23456]*.fits")))
    sci_frames_for_cube_C.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_0077[012]*.fits")))
    sci_frames_for_cube_C.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_00773[01234]*.fits")))

    sci_frames_for_cube_D = list(glob.glob(os.path.join(cookies_centered_06_directory, "*_00792[789]*.fits")))
    sci_frames_for_cube_D.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_0079[3456789]*.fits")))
    sci_frames_for_cube_D.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_00[89]*.fits")))
    sci_frames_for_cube_D.extend(glob.glob(os.path.join(cookies_centered_06_directory, "*_01*.fits")))
    '''
    
    # initialize and parallelize
    ## ## generalize the retrieved PCA vector cube as function of science frame range later!
    host_removal_fake_planets = HostRemoval(n_PCA = 100,
                                            outdir = config["data_dirs"]["DIR_FAKE_PSFS_HOST_REMOVED"], \
                                            abs_PCA_name = config["data_dirs"]["DIR_OTHER_FITS"] \
                                            + "pca_cubes_psfs/" \
                                            + "psf_PCA_vector_cookie_seqStart_004259_seqStop_005600.fits")
    '''
    host_removal_no_fake_planets_A = HostRemoval(n_PCA = 100,
                                            outdir = config["data_dirs"]["DIR_NO_FAKE_PSFS_HOST_REMOVED"], \
                                            abs_PCA_name = config["data_dirs"]["DIR_OTHER_FITS"] \
                                            + "pca_cubes_psfs/" \
                                            + "psf_PCA_vector_cookie_seqStart_004259_seqStop_005600.fits")

    host_removal_no_fake_planets_B = HostRemoval(n_PCA = 100,
                                            outdir = config["data_dirs"]["DIR_NO_FAKE_PSFS_HOST_REMOVED"], \
                                            abs_PCA_name = config["data_dirs"]["DIR_OTHER_FITS"] \
                                            + "pca_cubes_psfs/" \
                                            + "psf_PCA_vector_cookie_seqStart_006335_seqStop_006921.fits")

    host_removal_no_fake_planets_C = HostRemoval(n_PCA = 100,
                                            outdir = config["data_dirs"]["DIR_NO_FAKE_PSFS_HOST_REMOVED"], \
                                            abs_PCA_name = config["data_dirs"]["DIR_OTHER_FITS"] \
                                            + "pca_cubes_psfs/" \
                                            + "psf_PCA_vector_cookie_seqStart_007389_seqStop_007734.fits")

    host_removal_no_fake_planets_D = HostRemoval(n_PCA = 100,
                                            outdir = config["data_dirs"]["DIR_NO_FAKE_PSFS_HOST_REMOVED"], \
                                            abs_PCA_name = config["data_dirs"]["DIR_OTHER_FITS"] \
                                            + "pca_cubes_psfs/" \
                                            + "psf_PCA_vector_cookie_seqStart_008849_seqStop_009175.fits")
    '''
    
    # remove the host from the frames WITH fake planets
    #host_removal_fake_planets(fake_planet_frames_07_name_array[0])
    pool.map(host_removal_fake_planets, fake_planet_frames_07_name_array)

    '''
    # remove the host from the frames WITHOUT fake planets
    ## ## host_removal_no_fake_planets(cookies_centered_06_directory[0])
    pool.map(host_removal_no_fake_planets_A, sci_frames_for_cube_A)
    pool.map(host_removal_no_fake_planets_B, sci_frames_for_cube_B)
    pool.map(host_removal_no_fake_planets_C, sci_frames_for_cube_C)
    pool.map(host_removal_no_fake_planets_D, sci_frames_for_cube_D)
    '''
