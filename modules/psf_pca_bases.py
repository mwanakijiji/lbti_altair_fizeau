import multiprocessing
import configparser
import glob
import time
import pandas as pd
from astropy.io import fits
from astropy.convolution import convolve, Gaussian1DKernel, interpolate_replace_nans
from astropy.modeling import models, fitting
from modules import *

# import the PCA machinery for making backgrounds
from .basic_red import BackgroundPCACubeMaker

import matplotlib
matplotlib.use('agg') # avoids some crashes when multiprocessing
import matplotlib.pyplot as plt


class PSFPCACubeMaker:
    '''
    Make PCA bases of PSFs, in style of background PCA decomposition
    but with different sizes, masks (or no masks), no subtraction of background
    (i.e., it is assumed to already be subtracted away), and there is no
    consideration of channel variations

    INHERITANCE:
    BackgroundPCACubeMaker
    '''

    def __init__(self,
                 file_list,
                 n_PCA,
                 config_data = config,
                 subtract_median = True):
        '''
        INPUTS:
        file_list: list of ALL filenames in the directory
        n_PCA: number of PCA components to save in the cube
        -> (does NOT include possible separate components representing
        -> individual channel variations)
        config_data: configuration data, as usual
        subtract_median = True: the median frame will be subtracted from the
            cube before generating the basis set
        '''

        self.file_list = file_list
        self.n_PCA = n_PCA
        self.config_data = config_data
        self.subtract_median = subtract_median


    def __call__(self,
                 start_frame_num,
                 stop_frame_num,
                 resd_avg_limits,
                 x_gauss_limits,
                 y_gauss_limits,
                 write_training_cube_name,
                 write_median_frame_file_name,
                 write_abs_pca_cube_name):
        '''
        Make PCA cube (for future step of reconstructing) a PSF

        INPUTS:
        start_frame_num: starting frame number to use in PCA basis generation
        stop_frame_num: stopping [ditto], inclusive
        resd_avg_limits: a 2-element vector with the lower- and upper limits
            of the average of the residuals between PSF and Gaussian (acts as PSF quality criterion)
        x_gauss_limits: " " of the stdev in x of the best-fit Gaussian (acts as PSF quality criterion)
        y_gauss_limits: " " of the stdev in y of the best-fit Gaussian (acts as PSF quality criterion)
        write_training_cube_name: file under which to save the PCA training cube
        write_median_frame_file_name: file under which to save the median frame of the data which goes into
            the PCA training cube, BEFORE any median subtraction of that cube
        write_abs_pca_cube_name: file under which to save the PCA vector cube
        '''

        # read in a first file to get the shape
        test_img, header = fits.getdata(self.file_list[0], 0, header=True)
        shape_img = np.shape(test_img)

        print("psf_pca_bases: Initializing a PCA cube...")
        training_cube = np.nan*np.ones((stop_frame_num-start_frame_num+1,shape_img[0],shape_img[1]),
                                       dtype = np.float32)
        print("-"*prog_bar_width)

        # make the right mask (1=good; 0=masked)
        # (to make a circular mask, I made frames from stray code in phasecam_pupil_simulator.ipynb)
        '''
        Available masks:
        mask_406x406_rad080.fits
        mask_100x100_rad011.fits
        mask_100x100_rad021.fits
        mask_100x100_rad028.fits
        mask_100x100_ring_11_to_21.fits
        mask_100x100_ring_21_to_28.fits
        mask_100x100_rad_gtr_28.fits
        '''
        '''
        mask_weird, header = fits.getdata(self.config_data["data_dirs"]["DIR_OTHER_FITS"] + \
                                        "mask_406x406_rad080.fits", 0, header=True)
        mask_weird, header = fits.getdata(self.config_data["data_dirs"]["DIR_OTHER_FITS"] + \
                                        "mask_100x100_rad011.fits", 0, header=True)
        '''
        #import ipdb; ipdb.set_trace()
        #mask_weird = make_first_pass_mask(quad_choice)
        mask_weird = np.ones(shape_img) # no mask

        # initialize slice counter for removing unused slices later
        slice_counter = 0

        # loop over frames to add them to training cube
        for frame_num in range(start_frame_num, stop_frame_num+1):

            # get name of file that this number corresponds to
            abs_matching_file_array = [s for s in self.file_list if str("{:0>6d}".format(frame_num)) in s]

            # if there was a match
            if (len(abs_matching_file_array) != 0):

                # read in the science frame from raw data directory
                abs_matching_file = abs_matching_file_array[0] # get the name
                print("psf_pca_bases: Reading in frame "+str("{:0>6d}".format(frame_num)))
                print("  corresponding to file "+abs_matching_file)
                sci, header_sci = fits.getdata(abs_matching_file, 0, header=True)

                ## apply quality criteria for determining whether a frame should be added to a cube:
                # 1.) if the phase loop was closed
                #        (note this can include stand-alone closed-loop frames;
                # 2.) residuals are within limits
                # 3.) width of Gaussian fit is within limits
                if ((header_sci["PCCLOSED"] == 1) and
                    np.logical_and(header_sci["RESD_AVG"] >= resd_avg_limits[0],
                                   header_sci["RESD_AVG"] <= resd_avg_limits[1]) and
                    np.logical_and(header_sci["GAU_XSTD"] >= x_gauss_limits[0],
                                   header_sci["GAU_XSTD"] <= x_gauss_limits[1]) and
                    np.logical_and(header_sci["GAU_YSTD"] >= y_gauss_limits[0],
                                   header_sci["GAU_YSTD"] <= y_gauss_limits[1])):

                    # add to cube
                    training_cube[slice_counter,:,:] = sci

                    # advance counter
                    slice_counter += 1

                    # TEST only
                    print([frame_num,header_sci["RESD_AVG"],header_sci["GAU_XSTD"],header_sci["GAU_YSTD"]])

            # if there was no match
            elif (len(abs_matching_file_array) == 0):

                print("psf_pca_bases: Frame " + str("{:0>6d}".format(frame_num)) + " not found.")

            # if there were multiple matches
            else:

                print("psf_pca_bases: Something is amiss with your frame number choice.")
                break

        # remove the unused slices
        training_cube = training_cube[0:slice_counter,:,:]

        ## TEST: WRITE OUT
        '''
        hdu = fits.PrimaryHDU(training_cube)
        hdulist = fits.HDUList([hdu])
        hdu.writeto("junk.fits", clobber=True)
        '''
        ## END TEST

        # mask the raw training set
        training_cube_masked_weird = np.multiply(training_cube, mask_weird)

        # subtract the median from the training set
        median_frame = np.nanmedian(training_cube_masked_weird, axis = 0)
        if self.subtract_median:
            training_cube_masked_weird = np.subtract(training_cube_masked_weird, median_frame)

        # save the median for later use in reconstructing PSFs
        ''' REDUNDANT WITH BELOW
        fits.writeto(filename = raw_pca_training_median_name,
                     data = median_frame,
                     header = None,
                     overwrite = True)
        del median_frame
        print("psf_pca_bases: Wrote out raw PCA training cube median as \n " +
              raw_pca_training_median_name)
        print("-"*prog_bar_width)
        '''

        # write out training cube
        fits.writeto(filename = write_training_cube_name,
                     data = training_cube_masked_weird,
                     header = None,
                     overwrite = True)
        del training_cube
        print("psf_pca_bases: Wrote out PSF PCA training cube as \n " +
              write_training_cube_name +
              "\n with shape" +
              str(np.shape(training_cube_masked_weird)))
        print("-"*prog_bar_width)

        ## generate the PCA cube from the PSF data
        # first, generate and save the PCA offset frame (should be the median frame of the whole cube)
        print("psf_pca_bases: Writing median frame of PCA training cube out to \n" + write_median_frame_file_name)
        fits.writeto(filename = write_median_frame_file_name,
                     data = median_frame,
                     header = None,
                     overwrite = True)
        print("-"*prog_bar_width)

        # do the PCA decomposition
        pca_comp_cube = PCA_basis(training_cube_masked_weird, n_PCA = self.n_PCA)

        # write out the PCA vector cube
        fits.writeto(filename = write_abs_pca_cube_name,
                     data = pca_comp_cube,
                     header = None,
                     overwrite = True)
        print("psf_pca_bases: Wrote out PSF PCA vector cube as \n" +
              write_abs_pca_cube_name +
              "\n with shape" +
              str(np.shape(pca_comp_cube)))
        print("-"*prog_bar_width)


def main():
    '''
    Carry out steps to write out PSF PCA cubes
    '''

    # configuration data
    config = configparser.ConfigParser() # for parsing values in .init file
    config.read("modules/config.ini")

    # multiprocessing instance
    pool = multiprocessing.Pool(ncpu)

    # make a list of the Gaussian/centered PSF residual frames
    list_fits_residual_frame = list(glob.glob(str(config["data_dirs"]["DIR_CENTERED"] + "/*.fits")))
    print('list_resids')
    print(list_fits_residual_frame)

    ## initialize dataframe
    # frame_num: the LMIR frame number
    # resd_avg: the average absolute value of residuals
    # resd_med: the median " " "
    # resd_int: the integrated (i.e., summed) " " "
    df = pd.DataFrame(columns=["frame_num",
                               "resd_avg",
                               "resd_med",
                               "resd_int",
                               "x_gauss",
                               "y_gauss"])

    # file name to write residual data to
    residual_file_name = str(config["data_dirs"]["DIR_BIN"] +
                             config["file_names"]["RESID_CSV"])

    # initialize the file
    df.to_csv(residual_file_name)

    # lframe
    '''
    for q in range(0,len(list_fits_residual_frame)):
        print('Resid frame '+str(q))
        sciImg, header = fits.getdata(list_fits_residual_frame[q],0,header=True)
        print(list(header.keys()))
        # record frame number and residual values
        frame_num = int(list_fits_residual_frame[q].split(".")[-2].split("_")[-1])

        d = [{"frame_num": frame_num,
              "resd_avg": header["RESD_AVG"],
              "resd_med": header["RESD_MED"],
              "resd_int": header["RESD_INT"],
              "x_gauss": header["GAU_XSTD"],
              "y_gauss": header["GAU_YSTD"]}]

        d_df = pd.DataFrame(d)

        # append to file
        d_df.to_csv(residual_file_name, mode='a', header=False)

    print("psf_pca_bases: Residual data written to " + residual_file_name)
    '''
    # make a list of the centered cookie cutout files
    cookies_centered_06_directory = str(config["data_dirs"]["DIR_CENTERED"])
    cookies_centered_06_name_array = list(glob.glob(os.path.join(cookies_centered_06_directory, "*.fits")))

    ## generate PCA cubes for PSFs
    # (N.b. n_PCA needs to be smaller than the number of frames being used)

    # for PCA basis set for subtracting host star residuals
    pca_psf_maker_host_resids = PSFPCACubeMaker(file_list = cookies_centered_06_name_array,
                                    n_PCA = 100,
                                    subtract_median = True)
    # for PCA basis set for reconstructing host star (subtraction of median from the training
    # does not seem to make any difference when making a PCA basis set, though)
    pca_psf_maker_host_recon = PSFPCACubeMaker(file_list = cookies_centered_06_name_array,
                                    n_PCA = 100,
                                    subtract_median = False)

    # cube of fake data
    '''
    pca_psf_maker_host_resids(start_frame_num = 0,
                   stop_frame_num = 10000,
                   resd_avg_limits = [0, 0],
                   x_gauss_limits = [0, 0],
                   y_gauss_limits = [0, 0],
                   write_training_cube_name = str(config["data_dirs"]["DIR_OTHER_FITS"] +
                                                  'psf_PCA_training_cube_seqStart_00000_seqStop_10000_host_resids.fits'),
                    write_median_frame_file_name = str(config["data_dirs"]["DIR_PCA_CUBES_PSFS"] +
                                                       'median_frame_seqStart_00000_seqStop_10000_pcaNum_100_host_resids.fits'),
                    write_abs_pca_cube_name = str(config["data_dirs"]["DIR_PCA_CUBES_PSFS"] +
                                'psf_PCA_vector_cookie_seqStart_00000_seqStop_10000_pcaNum_100_host_resids.fits'))
    pca_psf_maker_host_recon(start_frame_num = 0,
                   stop_frame_num = 10000,
                   resd_avg_limits = [0, 0],
                   x_gauss_limits = [0, 0],
                   y_gauss_limits = [0, 0],
                   write_training_cube_name = str(config["data_dirs"]["DIR_OTHER_FITS"] +
                                                  'psf_PCA_training_cube_seqStart_00000_seqStop_10000_host_recon.fits'),
                    write_median_frame_file_name = str(config["data_dirs"]["DIR_PCA_CUBES_PSFS"] +
                                                       'median_frame_seqStart_00000_seqStop_10000_pcaNum_100_host_recon.fits'),
                    write_abs_pca_cube_name = str(config["data_dirs"]["DIR_PCA_CUBES_PSFS"] +
                                'psf_PCA_vector_cookie_seqStart_00000_seqStop_10000_pcaNum_100_host_recon.fits'))
    '''

    # cube A (sat)
    pca_psf_maker(start_frame_num = 4259,
                   stop_frame_num = 5608,
                   resd_avg_limits = [50,62.5],
                   x_gauss_limits = [4,6],
                   y_gauss_limits = [4,6],
                   write_training_cube_name = str(config["data_dirs"]["DIR_OTHER_FITS"] +
                   'psf_PCA_training_cube_seqStart_04259_seqStop_05608_host_resids.fits'),
                   write_median_frame_file_name = str(config["data_dirs"]["DIR_PCA_CUBES_PSFS"] +
                   'median_frame_seqStart_04259_seqStop_05608_pcaNum_100_host_resids.fits'),
                   write_abs_pca_cube_name = str(config["data_dirs"]["DIR_PCA_CUBES_PSFS"] +
                   'psf_PCA_vector_cookie_seqStart_04259_seqStop_05608_pcaNum_100_host_resids.fits'))

    # cube B (unsat)
    pca_psf_maker(start_frame_num = 6303,
                   stop_frame_num = 6921,
                   resd_avg_limits = [35.3,37.3],
                   x_gauss_limits = [3.9,6.7],
                   y_gauss_limits = [3.9,6.7],
                   write_training_cube_name = str(config["data_dirs"]["DIR_OTHER_FITS"] +
                   'psf_PCA_training_cube_seqStart_06303_seqStop_06921_host_resids.fits'),
                   write_median_frame_file_name = str(config["data_dirs"]["DIR_PCA_CUBES_PSFS"] +
                   'median_frame_seqStart_06303_seqStop_06921_pcaNum_100_host_resids.fits'),
                   write_abs_pca_cube_name = str(config["data_dirs"]["DIR_PCA_CUBES_PSFS"] +
                   'psf_PCA_vector_cookie_seqStart_06303_seqStop_06921_pcaNum_100_host_resids.fits'))
    # cube C (unsat)
    pca_psf_maker(start_frame_num = 7120,
                   stop_frame_num = 7734,
                   resd_avg_limits = [35.4,40.6],
                   x_gauss_limits = [3.9,4.5],
                   y_gauss_limits = [3.9,4.5],
                   write_training_cube_name = str(config["data_dirs"]["DIR_OTHER_FITS"] +
                   'psf_PCA_training_cube_seqStart_07120_seqStop_07734_host_resids.fits'),
                   write_median_frame_file_name = str(config["data_dirs"]["DIR_PCA_CUBES_PSFS"] +
                   'median_frame_seqStart_07120_seqStop_07734_pcaNum_100_host_resids.fits'),
                   write_abs_pca_cube_name = str(config["data_dirs"]["DIR_PCA_CUBES_PSFS"] +
                   'psf_PCA_vector_cookie_seqStart_07120_seqStop_07734_pcaNum_100_host_resids.fits'))
    # cube D (sat)
    pca_psf_maker(start_frame_num = 7927,
                   stop_frame_num = 11408,
                   resd_avg_limits = [40.6,55],
                   x_gauss_limits = [4.15,5],
                   y_gauss_limits = [4.1,4.44],
                   write_training_cube_name = str(config["data_dirs"]["DIR_OTHER_FITS"] +
                   'psf_PCA_training_cube_seqStart_07927_seqStop_11408_host_resids.fits'),
                   write_median_frame_file_name = str(config["data_dirs"]["DIR_PCA_CUBES_PSFS"] +
                   'median_frame_seqStart_07927_seqStop_11408_pcaNum_100_host_resids.fits'),
                   write_abs_pca_cube_name = str(config["data_dirs"]["DIR_PCA_CUBES_PSFS"] +
                   'psf_PCA_vector_cookie_seqStart_07927_seqStop_11408_pcaNum_100_host_resids.fits'))
