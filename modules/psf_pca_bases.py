import multiprocessing
import configparser
import glob
import time
import sys
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
                 unique_read_string,
                 unique_write_string):
        '''
        Make PCA cube (for future step of reconstructing) a PSF

        INPUTS:
        start_frame_num: starting frame number to use in PCA basis generation
        stop_frame_num: stopping [ditto], inclusive
        resd_avg_limits: a 2-element vector with the lower- and upper limits
            of the average of the residuals between PSF and Gaussian (acts as PSF quality criterion)
        x_gauss_limits: " " of the stdev in x of the best-fit Gaussian (acts as PSF quality criterion)
        y_gauss_limits: " " of the stdev in y of the best-fit Gaussian (acts as PSF quality criterion)
        unique_read_string: unique string (immediately before file extension) for reading in files
        unique_write_string: unique string (immediately before file extension) for writing out files
        '''

        # read in a first file to get the shape
        test_img, header = fits.getdata(self.file_list[0], 0, header=True)
        shape_img = np.shape(test_img)

        print("psf_pca_bases: Initializing a PCA cube... \n" + prog_bar_width*"-")
        training_cube = np.nan*np.ones((stop_frame_num-start_frame_num+1,shape_img[0],shape_img[1]),
                                       dtype = np.float32)

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
        print("psf_pca_bases: Adding frames to PCA training cube \n" + prog_bar_width*"-")
        for frame_num in range(start_frame_num, stop_frame_num+1):

            # get name of file that this number corresponds to
            abs_matching_file_array = [s for s in self.file_list if str("{:0>6d}".format(frame_num)) in s]

            # if there was a match
            if (len(abs_matching_file_array) != 0):

                #print("psf_pca_bases: Reading in frame "+str("{:0>6d}".format(frame_num)))

                # read in the science frame from raw data directory
                abs_matching_file = abs_matching_file_array[0] # get the name
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
                    #print([frame_num,header_sci["RESD_AVG"],header_sci["GAU_XSTD"],header_sci["GAU_YSTD"]])

            # if there was no match
            elif (len(abs_matching_file_array) == 0):

                print("\rpsf_pca_bases: Frame " + str("{:0>6d}".format(frame_num)) + " not found.")

            # if there were multiple matches
            else:

                print("psf_pca_bases: Something is amiss with your frame number choice.")
                break

            # update progress bar
            n = int((prog_bar_width+1)* (frame_num-start_frame_num) / np.subtract(stop_frame_num,start_frame_num))
            sys.stdout.write("\r[{0}{1}]".format("#" * n, " " * (prog_bar_width - n)))

        print("\n") # space
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

        if self.subtract_median:
            # subtract the median from the training set
            median_frame = np.nanmedian(training_cube_masked_weird, axis = 0)
            training_cube_masked_weird = np.subtract(training_cube_masked_weird, median_frame)
        
        training_cube_name = str(self.config_data["data_dirs"]["DIR_OTHER_FITS"] +
                                'psf_PCA_training_cube' +
                                '_seqStart_'+str("{:0>6d}".format(start_frame_num)) +
                                '_seqStop_'+str("{:0>6d}".format(stop_frame_num)) +
                                unique_write_string + '.fits')
        fits.writeto(filename = training_cube_name,
                     data = training_cube_masked_weird,
                     header = None,
                     overwrite = True)
        del training_cube
        
        print("psf_pca_bases: Wrote out PSF PCA training cube as \n" +
              training_cube_name + "\n" +
              prog_bar_width*"-")

        ## generate the PCA cube from the PSF data
        # first, generate and save the PCA offset frame (should be the median frame of the whole cube)
        median_frame_file_name = str(self.config_data["data_dirs"]["DIR_PCA_CUBES_PSFS"] +
                                'median_frame_seqStart_' + str("{:0>6d}".format(start_frame_num)) +
                                '_seqStop_' + str("{:0>6d}".format(stop_frame_num)) + '_pcaNum_'
                                + str("{:0>4d}".format(self.n_PCA)) + unique_write_string + '.fits')
        fits.writeto(filename = median_frame_file_name,
                     data = median_frame,
                     header = None,
                     overwrite = True)
        print("psf_pca_bases: Wrote median frame of PCA training cube out to \n" +
              median_frame_file_name + "\n" +
              prog_bar_width*"-")

        # do the PCA decomposition
        pca_comp_cube = PCA_basis(training_cube_masked_weird, n_PCA = self.n_PCA)

        # write out the PCA vector cube
        abs_pca_cube_name = str(self.config_data["data_dirs"]["DIR_PCA_CUBES_PSFS"] +
                                'psf_PCA_vector_cookie' +
                                '_seqStart_'+str("{:0>6d}".format(start_frame_num))+
                                '_seqStop_'+str("{:0>6d}".format(stop_frame_num))+'_pcaNum_'
                                +str("{:0>4d}".format(self.n_PCA)) + unique_write_string + '.fits')
        fits.writeto(filename = abs_pca_cube_name,
                     data = pca_comp_cube,
                     header = None,
                     overwrite = True)
        print("psf_pca_bases: Wrote out PSF PCA vector cube as \n" +
              abs_pca_cube_name + "\n" +
              prog_bar_width*"-")


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
    #print('list_resids')
    #print(list_fits_residual_frame)

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

    # populate dataframe
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

    print("Residual data written to " + residual_file_name)
    '''
    # make a list of the centered cookie cutout files
    cookies_centered_06_directory = str(config["data_dirs"]["DIR_CENTERED"])
    cookies_centered_06_name_array = list(glob.glob(os.path.join(cookies_centered_06_directory, "*.fits")))

    # generate PCA cubes for PSFs
    # (N.b. n_PCA needs to be smaller than the number of frames being used)

    # cube for subtracting the host star
    # (median value must be subtracted since host star residuals-relative-to-the-median
    # will be subtracted from frames)
    pca_psf_maker_subt_host = PSFPCACubeMaker(file_list = cookies_centered_06_name_array,
                                    n_PCA = 100,
                                    subtract_median = True)

    # cube for reconstructing the full host star PSF
    # (median value is NOT subtracted since this is for making fake planet PSFs without
    # saturation effects, and determining the host star amplitude)
    pca_psf_maker_recon_host = PSFPCACubeMaker(file_list = cookies_centered_06_name_array,
                                    n_PCA = 100,
                                    subtract_median = False)
    # cube A
    '''
    pca_psf_maker(start_frame_num = 4259,
                   stop_frame_num = 5608,
                   resd_avg_limits = [50,62.5],
                   x_gauss_limits = [4,6],
                   y_gauss_limits = [4,6])

    # cube B (unsat)
    pca_psf_maker(start_frame_num = 6303,
                   stop_frame_num = 6921,
                   resd_avg_limits = [35.3,37.3],
                   x_gauss_limits = [3.9,6.7],
                   y_gauss_limits = [3.9,6.7])
    # cube C (unsat)
    pca_psf_maker(start_frame_num = 7120,
                   stop_frame_num = 7734,
                   resd_avg_limits = [35.4,40.6],
                   x_gauss_limits = [3.9,4.5],
                   y_gauss_limits = [3.9,4.5])
    # cube D 
    pca_psf_maker(start_frame_num = 7927,
                   stop_frame_num = 11408,
                   resd_avg_limits = [40.6,55],
                   x_gauss_limits = [4.15,5],
                   y_gauss_limits = [4.1,4.44])
    '''

    # make cube for subtracting host star
    pca_psf_maker_subt_host(start_frame_num = 0,
                   stop_frame_num = 10000,
                   resd_avg_limits = [0, 0],
                   x_gauss_limits = [0, 0],
                   y_gauss_limits = [0, 0],
                   unique_read_string = " ",
                   unique_write_string = "_for_subt_host_synth")

    # make cube for reconstructing full PSF
    pca_psf_maker_recon_host(start_frame_num = 0,
                   stop_frame_num = 10000,
                   resd_avg_limits = [0, 0],
                   x_gauss_limits = [0, 0],
                   y_gauss_limits = [0, 0],
                   unique_read_string = " ",
                   unique_write_string = "_for_recon_host_synth") 


