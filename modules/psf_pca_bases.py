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
                 config_data = config):
        '''
        INPUTS:
        file_list: list of ALL filenames in the directory
        n_PCA: number of PCA components to save in the cube
        -> (does NOT include possible separate components representing
        -> individual channel variations)
        config_data: configuration data, as usual
        '''

        self.file_list = file_list
        self.n_PCA = n_PCA
        self.config_data = config_data


    def __call__(self,
                 start_frame_num,
                 stop_frame_num):
        '''
        Make PCA cube (for future step of reconstructing) a PSF

        INPUTS:
        start_frame_num: starting frame number to use in PCA basis generation
        stop_frame_num: stopping [ditto], inclusive
        '''

        # read in a first file to get the shape
        test_img, header = fits.getdata(self.file_list[0], 0, header=True)
        shape_img = np.shape(test_img)

        print("Initializing a PCA cube...")
        training_cube = np.nan*np.ones((stop_frame_num-start_frame_num+1,shape_img[0],shape_img[1]),
                                       dtype = np.float32)

        # make the right mask (1=good; 0=masked)
        # (to make a circular mask, I made frames from stray code in phasecam_pupil_simulator.ipynb)
        mask_weird, header = fits.getdata(self.config_data["data_dirs"]["DIR_OTHER_FITS"] + \
                                        "mask_406x406_rad080.fits", 0, header=True)
        #mask_weird = make_first_pass_mask(quad_choice) 
        #mask_weird = np.ones(shape_img) # no mask

        # initialize slice counter for removing unused slices later
        slice_counter = 0

        # loop over frames to add them to training cube
        for frame_num in range(start_frame_num, stop_frame_num+1):

            # get name of file that this number corresponds to
            abs_matching_file_array = [s for s in self.file_list if str("{:0>6d}".format(frame_num)) in s]

            # if there was a match
            if (len(abs_matching_file_array) != 0):

                print("Reading in frame "+str("{:0>6d}".format(frame_num)))

                # read in the science frame from raw data directory
                abs_matching_file = abs_matching_file_array[0] # get the name
                sci, header_sci = fits.getdata(abs_matching_file, 0, header=True)

                ## apply criteria for determining whether a frame should be added to a cube:
                # 1.) if the phase loop was closed
                #        (note this can include stand-alone closed-loop frames;
                # 2.) residuals between ...
                # 3.) width of Gaussian fit is between ...
                # may need to refine this criterion later)
                ## ## ADD ANOTHER CRITERION BASED ON RESIDUALS WITH GAUSSIAN FIT
                ## ## (I.E., READ IN CSV FILE POPULATED WITH RESID LEVELS)
                if (header_sci["PCCLOSED"] == 1):

                    # add to cube
                    training_cube[slice_counter,:,:] = sci

                    # advance counter
                    slice_counter += 1

            # if there was no match
            elif (len(abs_matching_file_array) == 0):

                print("Frame " + str("{:0>6d}".format(frame_num)) + " not found.")

            # if there were multiple matches
            else:

                print("Something is amiss with your frame number choice.")
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
        training_cube_name = str(self.config_data["data_dirs"]["DIR_OTHER_FITS"] +
                                'psf_PCA_training_cube' +
                                '_seqStart_'+str("{:0>6d}".format(start_frame_num))+
                                '_seqStop_'+str("{:0>6d}".format(stop_frame_num))+'.fits')
        fits.writeto(filename=training_cube_name,
                     data=training_cube_masked_weird,
                     header=None,
                     overwrite=True)
        del training_cube
        print("Wrote out PSF PCA training cube " + os.path.basename(training_cube_name) + ", with shape")
        print(training_cube_name)
        print(np.shape(training_cube_masked_weird))

        # generate the PCA cube from the PSF data
        pca_comp_cube = PCA_basis(training_cube_masked_weird, n_PCA = self.n_PCA)

        # write out the PCA vector cube
        abs_pca_cube_name = str(self.config_data["data_dirs"]["DIR_PCA_CUBES_PSFS"] +
                                'psf_PCA_vector_cookie' +
                                '_seqStart_'+str("{:0>6d}".format(start_frame_num))+
                                '_seqStop_'+str("{:0>6d}".format(stop_frame_num))+'.fits')
        fits.writeto(filename=abs_pca_cube_name,
                     data=pca_comp_cube,
                     header=None,
                     overwrite=True)
        print("Wrote out PSF PCA vector cube " + os.path.basename(abs_pca_cube_name) + ", with shape")
        print(np.shape(pca_comp_cube))
        print("---------------------------")


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

    # populate dataframe
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
    
    # make a list of the centered cookie cutout files
    cookies_centered_06_directory = str(config["data_dirs"]["DIR_CENTERED"])
    cookies_centered_06_name_array = list(glob.glob(os.path.join(cookies_centered_06_directory, "*.fits")))

    # generate PCA cubes for PSFs
    # (N.b. n_PCA needs to be smaller than the number of frames being used)
    ## WILL NEED MORE PARAMETER ARRAYS FOR VARIOUS FRAME SEQUENCES
    pca_psf_maker = PSFPCACubeMaker(file_list = cookies_centered_06_name_array,
                                    n_PCA = 100) # create instance
    # cube A
    pca_psf_maker(start_frame_num = 4259,
                   stop_frame_num = 5600)
    # cube B (unsat)
    pca_psf_maker(start_frame_num = 6335,
                   stop_frame_num = 6921)
    # cube C (unsat)
    pca_psf_maker(start_frame_num = 7389,
                   stop_frame_num = 7734)
    # cube D 
    pca_psf_maker(start_frame_num = 8849,
                   stop_frame_num = 9175)


