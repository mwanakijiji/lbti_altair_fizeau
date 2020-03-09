import multiprocessing
import configparser
import glob
import time
import pickle
import math
import datetime
import matplotlib.pyplot as plt
import pandas as pd
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
                 abs_PCA_training_median_name,
                 config_data = config):
        '''
        INPUTS:
        n_PCA: number of principal components to use
        outdir: directory to deposit the host-subtracted images in (this has to be
                       defined at the function call because the images may or may not
                       contain fake planet PSFs, and I want to keep them separate)
        abs_PCA_name: absolute file name of the PCA cube to reconstruct the host star
                       for making a fake planet (i.e., without saturation effects)
        abs_PCA_training_median_name: absolute file name of the median of the PCA cube training set,
                       which was subtracted before the PCA basis was made; this frame will also
                       be subtracted from the science frame
        config_data: configuration data, as usual
        '''

        self.n_PCA = n_PCA
        self.outdir = outdir
        self.abs_PCA_name = abs_PCA_name
        self.abs_PCA_training_median_name = abs_PCA_training_median_name
        self.config_data = config_data

        # read in the median of the PCA training cube before this median was subtracted from that cube and the cube was decomposed
        self.abs_PCA_training_median, self.header_abs_PCA_training_median = fits.getdata(self.abs_PCA_training_median_name, 0, header=True)

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

        print("host_removal: "+str(datetime.datetime.now())+" Read in PCA training median \n" + self.abs_PCA_training_median_name)
        print("-"*prog_bar_width)
        print("host_removal: "+str(datetime.datetime.now())+" Read in PCA basis cube \n" + self.abs_PCA_name)
        print("-"*prog_bar_width)
        print("host_removal: "+str(datetime.datetime.now())+" Reading in science frame \n" + abs_sci_name)
        print("-"*prog_bar_width)

        # read in the cutout science frame
        # (there should be no masking of this frame downstream)
        sci, header_sci = fits.getdata(abs_sci_name, 0, header=True)

        # subtract the median of the PCA training set
        sci = np.subtract(sci, self.abs_PCA_training_median)

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
            fit_unsat = fit_pca_star(pca_cube = self.pca_basis_cube_sat,
                                     sciImg = sci,
                                     raw_pca_training_median = np.nan*np.ones(np.shape(self.pca_basis_cube_host_star[0,:,:])),
                                     mask_weird = no_mask,
                                     n_PCA=100,
                                     subt_median = False)
        except:
            return
        #import ipdb; ipdb.set_trace()
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
        print("host_removal: "+str(datetime.datetime.now())+" Writing out host_removed frame \n"
              + abs_image_host_removed_name)
        print("-"*prog_bar_width)


class HostRemovalCube:
    '''
    PCA-decompose a saturated host star PSF and remove it, using a cube of frames in memory
    N.b. This does no de-rotation; is blind to parallactic angle
    '''

    def __init__(self,
                injection_iteration,
                 fake_params,
                 cube_frames,
                 n_PCA,
                 outdir,
                 write_name_abs_host_star_PCA,
                 abs_fake_planet_PCA_name,
                 abs_region_mask_name,
                 frame_array,
                 config_data = config,
                 subtract_median_PCA_training_frame = True,
                 write = False):
        '''
        INPUTS:
        injection_iteration: iteration of fake planet injection (None: no fake planet)
        fake_params: fake planet parameters (if applicable; this is just for making the
            file name string if we are writing out to disk; if not applicable, but
            in some other size-3 Pandas DataFrame of strings)
        cube_frames: the cube of frames to host-subtract, before host-star subtraction
        n_PCA: number of principal components to use
        outdir: directory to deposit the host-subtracted images in (this has to be
                       defined at the function call because the images may or may not
                       contain fake planet PSFs, and I want to keep them separate)
        write_name_abs_host_star_PCA: absolute file name of the PCA cube to reconstruct the host
                       star for host star subtraction (i.e., this is probably with saturation)
        abs_fake_planet_PCA_name: absolute file name of the PCA cube to reconstruct the host star
                       to make a fake planet (i.e., without saturation effects)
        abs_region_mask_name: absolute file name of the mask to apply when doing PCA; this will be
                       a cube, where each slice defines a region to PCA-decompose in turn (one
                       single frame would mean just one region is being used for the PCA decomposition);
                       the individual PCA reconstructions are combined into the final image
        frame_array: array of integers corresponding to the frame file name numbers
        config_data: configuration data, as usual
        subtract_median_PCA_training_frame = True: subtract from the science frames the median frame of
            the raw PCA training set which went into the generation of the PCA basis set (because
            just before that PCA basis set was generated, the median was subtracted from that
            training set)
        write = True: data product should be written to disk (for checking)

        (REMOVED:)
        classical_ADI: this just subtracts a median of the whole cube from each slice,
            as opposed to doing a subtraction individualized to each slice; note that True
            also means that the PCA cubes that are read in are ignored (default False)
        '''

        self.injection_iteration = injection_iteration
        self.fake_params = fake_params
        self.cube_frames = cube_frames
        self.n_PCA = n_PCA
        self.outdir = outdir
        self.abs_host_star_PCA_name = write_name_abs_host_star_PCA
        self.abs_fake_planet_PCA_name = abs_fake_planet_PCA_name
        self.abs_region_mask_name = abs_region_mask_name
        self.frame_num_array = frame_array
        self.config_data = config_data
        self.subtract_median_PCA_training_frame = subtract_median_PCA_training_frame
        self.write = write

        # read in the PCA vector cube for this series of frames
        # (note the PCA needs to correspond to saturated PSFs, since I am subtracting
        # saturated PSFs away)
        self.pca_basis_cube_host_star, self.header_pca_basis_cube_host_star = fits.getdata(self.abs_host_star_PCA_name, 0, header=True)
        self.pca_basis_cube_fake_planet, self.header_pca_basis_fake_planet = fits.getdata(self.abs_fake_planet_PCA_name, 0, header=True)

        # read in the mask
        self.abs_region_mask, self.header_abs_region_mask = fits.getdata(self.abs_region_mask_name, 0, header=True)

        ##########


    def __call__(self):
        '''
        Reconstruct and subtract the host star from each slice

        INPUTS:
        abs_sci_name: the absolute path of the science frame into which we want to inject a planet

        OUTPUTS:
        host_subt_cube: a cube of non-derotated frames
        '''
        print("host_removal: "+str(datetime.datetime.now())+" Read in PCA basis for host star \n" + self.abs_host_star_PCA_name)
        print("-"*prog_bar_width)
        print("host_removal: "+str(datetime.datetime.now())+" Read in PCA basis for fake planet \n" + self.abs_fake_planet_PCA_name)
        print("-"*prog_bar_width)
        print("host_removal: "+str(datetime.datetime.now())+" Read in PCA tesselation file \n" + self.abs_region_mask_name)
        print("-"*prog_bar_width)

        # string for making subdirectories to place ADI frames in
        if self.injection_iteration:
            injection_iteration_string = "inj_iter_" + str(self.injection_iteration).zfill(4)
        else:
            injection_iteration_string = "no_fake_planet"

        # make a cube that is the same shape as the input
        host_subt_cube_all_frames = np.nan*np.ones(np.shape(self.cube_frames))
        # make a cube for storing images of the reconstructed PSFs, for checking
        recon_frames_cube_all_frames = np.copy(host_subt_cube_all_frames)

        # subtract the same median which was subtracted from the PCA training set
        if self.subtract_median_PCA_training_frame:
            print('')

        # subtract the median PCA training set frame before decomposition
        if False:
            print('') # this is vestigial

        # or else do PCA
        else:

            # remove the host star from each slice
            for slice_num in range(0,len(self.cube_frames)):

                # initialize cube to hold reconstructed regions of each slice
                cube_PCA_recon_regions_1_frame = np.zeros(np.shape(self.abs_region_mask))
                # ... and initialize cube to hold host-star subtracted regions
                cube_host_subt_regions_1_frame = np.copy(cube_PCA_recon_regions_1_frame)
                # ... and also an FYI cube to hold regions of the original image
                cube_original_image_1_frame = np.copy(cube_PCA_recon_regions_1_frame)

                print("host_removal: "+str(datetime.datetime.now())+\
                    " Removing host star from relative slice " + str(slice_num) +\
                    " of " + str(len(self.cube_frames)))
                print("-"*prog_bar_width)

                # select the slice from the cube
                # (there should be no masking of this frame downstream)
                sci = self.cube_frames[slice_num,:,:]

                # define the mask for weird pixels of this science frame
                # 1= good; np.nan= masked
                ## ## fine-tune this step later!
                mask_weird_pixels = np.ones(np.shape(sci)) # initialize
                no_mask = np.copy(mask_weird_pixels) # a non-mask for reconstructing saturated PSFs

                # mask weird pixels
                '''
                mask_weird_predefined, header = fits.getdata(self.config_data["data_dirs"]["DIR_OTHER_FITS"] + \
                                                             "mask_100x100_rad_gtr_28.fits", 0, header=True)
                mask_weird_pixels[mask_weird_predefined == 0] = np.nan
                '''
                # end predefined mask section

                # mask based on saturation level
                # mask_weird[sci > 1e8] = np.nan # mask saturating region

                # mask based on region for the PCA reconstruction;
                # loop over each region
                for mask_slice_num in range(0,len(self.abs_region_mask)):

                    # slice defining this region
                    this_region = self.abs_region_mask[mask_slice_num,:,:]

                    # change value convention to fit the PCA decomposition
                    # 1= good; np.nan= masked
                    nan_mask = np.nan*np.ones(np.shape(this_region))
                    nan_mask[this_region != 0] = 1
                    this_region = this_region*nan_mask
                    # at this point, this_region should have
                    # 1. 1s inside the region of interest, so as to define it
                    # 2. nans outside the region of interest


                    ## combine the region mask with the weird pixel mask

                    # add the masks, and convert all non-nan values to 1
                    mask_for_region_and_weird_pixels = np.add(this_region, mask_weird_pixels)
                    mask_for_region_and_weird_pixels[np.isfinite(mask_for_region_and_weird_pixels)] = 1

                    ## TEST: WRITE OUT
                    '''
                    hdu = fits.PrimaryHDU(mask_for_region_and_weird_pixels)
                    hdulist = fits.HDUList([hdu])
                    hdu.writeto("junk_mask_for_region_and_weird_pixels_"+str(mask_slice_num)+".fits", clobber=True)

                    hdu = fits.PrimaryHDU(this_region)
                    hdulist = fits.HDUList([hdu])
                    hdu.writeto("junk_this_region_"+str(mask_slice_num)+".fits", clobber=True)
                    '''
                    ## END TEST

                    ###########################################
                    # PCA-decompose the host star PSF
                    # (note no de-rotation of the image here)

                    # do the PCA fit of masked host star
                    # returns dict: 'pca_vector': the PCA best-fit vector; and 'recon_2d': the 2D reconstructed PSF
                    # N.b. PCA reconstruction will be to get an UN-sat PSF; note PCA basis cube involves unsat PSFs

                    try:
                        # fit to the host star for subtraction, within the region corresponding to this mask slice
                        fit_host_star = fit_pca_star(pca_cube = self.pca_basis_cube_host_star,
                                                 sciImg = sci,
                                                 raw_pca_training_median = np.nan*np.ones(np.shape(self.pca_basis_cube_host_star[0,:,:])),
                                                 mask_weird = mask_for_region_and_weird_pixels,
                                                 n_PCA = self.n_PCA,
                                                 subt_median = False)

                    except:
                        print("host_removal: "+str(datetime.datetime.now())+" PCA fit to slice number " + str(slice_num) + " failed; skipping.")
                        continue
                    #import ipdb; ipdb.set_trace()
                    # subtract the PCA-reconstructed host star (within the region corresponding to this mask slice)
                    region_host_removed = np.subtract(np.multiply(sci,self.abs_region_mask[mask_slice_num,:,:]),
                                                      fit_host_star["recon_2d_masked"])

                    # put the reconstructed region into the cube
                    cube_PCA_recon_regions_1_frame[mask_slice_num,:,:] = fit_host_star["recon_2d_masked"]

                    # put the host-star-subtracted region its cube
                    cube_host_subt_regions_1_frame[mask_slice_num,:,:] = region_host_removed

                    # put the region of the original image into its cube
                    cube_original_image_1_frame[mask_slice_num,:,:] = np.multiply(sci,self.abs_region_mask[mask_slice_num,:,:])

                    # save the PCA info as a text file (FYI only)
                    csv_file_name = str(self.config_data["data_dirs"]["DIR_FYI_INFO"]) + \
                        "pcaspectrumsciencecubeframe_" + str(slice_num).zfill(6) + \
                        "_maskslice_" + str(mask_slice_num).zfill(4) + \
                        "_fakeangledegEofN_" + str(self.fake_params["angle_deg_EofN"]) + \
                        "_fakeradasec_" + str(self.fake_params["rad_asec"]) + \
                        "_fakeampllinearnorm_" + str(self.fake_params["ampl_linear_norm"]) + \
                        "_fakeampllinearnorm0_" + str(self.fake_params["ampl_linear_norm"]) + \
                        ".csv"
                    pca_vec_write = pd.DataFrame(data=soln_vector[0],columns=["amplitude"])
                    pca_vec_write.to_csv(csv_file_name, sep = ",")
                    # add some meta-data to the bottom of the file
                    # (I have not found an easy way to insert it at the top)
                    file_object = open(csv_file_name, 'a')
                    file_object.write('-------------------\n')
                    file_object.write('self.injection_iteration: ' + \
                        str(self.injection_iteration) + '\n')
                    file_object.write('cube slice_num: ' + \
                        str(slice_num).zfill(6) + '\n')
                    file_object.write('self.fake_params: ' + \
                        str(self.fake_params) + '\n')
                    file_object.write('self.n_PCA: ' + \
                        str(self.n_PCA) + '\n')
                    file_object.write('self.outdir: ' + \
                        str(self.outdir) + '\n')
                    file_object.write('self.abs_host_star_PCA_name: ' + \
                        str(self.abs_host_star_PCA_name) + '\n')
                    file_object.write('self.abs_fake_planet_PCA_name: ' + \
                        str(self.abs_fake_planet_PCA_name) + '\n')
                    file_object.write('self.abs_region_mask_name: ' + \
                        str(self.abs_region_mask_name) + '\n')
                    file_object.write('mask slice num: ' + \
                        str(mask_slice_num).zfill(4))
                    file_object.close()
                    print("host_removal: Wrote PCA spectrum file " + csv_file_name)

                    # accumulate-plot the PCA vectors (FYI only)
                    plt.plot(fit_host_star["pca_vector"],
                        label="tess. reg. "+str(mask_slice_num)) # this will be overplotted
                    # if we're at the last region to plot the PCA vector of
                    if mask_slice_num == len(self.abs_region_mask)-1:
                        plot_file_name = str(self.config_data["data_dirs"]["DIR_FYI_INFO"]) + \
                          "pca_spectrum_science_cube_frame_" + \
                          str(slice_num).zfill(6) + \
                          "_mask_slice_"+str(mask_slice_num).zfill(4) + ".pdf"
                        plt.legend(loc="upper right")
                        plt.xlabel("PCA mode")
                        plt.ylabel("Amplitude")
                        plt.savefig(plot_file_name)
                        plt.clf()
                        print("host_removal: Plotted PCA vectors in \n" + plot_file_name)

                    ## BEGIN TEST
                    '''
                        fits.writeto(filename = "junk_host_removed_"+str(slice_num)+".fits",
								 data = image_host_removed,
								 overwrite = True)
                        fits.writeto(filename = "junk_sci_"+str(slice_num)+".fits",
								 data = sci,
								 overwrite = True)
                        fits.writeto(filename = "junk_recon_fit_"+str(slice_num)+".fits",
								 data = fit_host_star["recon_2d"],
								 overwrite = True)
                        print("WROTE TEST FILES")
                    '''
                    ## END TEST

                ## TEST: WRITE OUT
                # the raw regions
                '''
                hdu = fits.PrimaryHDU(cube_original_image_1_frame)
                hdulist = fits.HDUList([hdu])
                hdu.writeto("junk_cube_original_image_1_frame_"+str(slice_num)+".fits", clobber=True)

                # reconstructed regions
                hdu = fits.PrimaryHDU(cube_PCA_recon_regions_1_frame)
                hdulist = fits.HDUList([hdu])
                hdu.writeto("junk_cube_PCA_recon_regions_1_frame_"+str(slice_num)+".fits", clobber=True)

                # host star subtracted regions
                hdu = fits.PrimaryHDU(cube_host_subt_regions_1_frame)
                hdulist = fits.HDUList([hdu])
                hdu.writeto("junk_cube_host_subt_regions_1_frame_"+str(slice_num)+".fits", clobber=True)
                '''
                ## END TEST

                # anything that is still zero in the recon regions cube, turn it to nan
                cube_PCA_recon_regions_1_frame[cube_PCA_recon_regions_1_frame == 0] = np.nan
                # take median of reconstructed frame
                # (this represents 1 science readout)
                final_PCA_recon_frame = np.nanmedian(cube_PCA_recon_regions_1_frame, axis=0)

                # combine the host-star subtracted regions into one frame
                # (again, this represents 1 science readout)
                final_host_subt_frame = np.nanmedian(cube_host_subt_regions_1_frame, axis=0)

                # put the reconstructed PSF into the larger cube of all PSFs
                recon_frames_cube_all_frames[slice_num,:,:] = final_PCA_recon_frame

                # and put the host-star-subtracted image into the larger cube of all readouts
                host_subt_cube_all_frames[slice_num,:,:] = final_host_subt_frame

                ## TEST: WRITE OUT
                '''
                hdu = fits.PrimaryHDU(final_PCA_recon_frame)
                hdulist = fits.HDUList([hdu])
                hdu.writeto("junk_final_PCA_recon_frame_"+str(slice_num)+".fits", overwrite=True)
                #print(": Wrote junk_final_PCA_recon_frame_"+str(slice_num)+".fits as test")

                hdu = fits.PrimaryHDU(final_host_subt_frame)
                hdulist = fits.HDUList([hdu])
                hdu.writeto("junk_final_host_subt_frame_"+str(slice_num)+".fits", overwrite=True)
                '''
                ## END TEST

        # if writing to disk for checking
        if self.write:

            # check if injection_iteration_string directory exists; if not, make the directory
            abs_path_name = self.config_data["data_dirs"]["DIR_OTHER_FITS"] + \
                                        injection_iteration_string + "/"
            if not os.path.exists(abs_path_name):
                os.makedirs(abs_path_name)
                print("host_removal: "+str(datetime.datetime.now())+" Made directory " + abs_path_name)

            # the cube of frames which are going to be PCA-reconstructed
            file_name_to_recon = self.config_data["data_dirs"]["DIR_OTHER_FITS"] + \
                injection_iteration_string + "/" + \
                "cube_to_pca_recon_" + \
                str(self.fake_params["angle_deg_EofN"]) + \
                "_" + str(self.fake_params["rad_asec"]) + \
                "_" + str(self.fake_params["ampl_linear_norm"]) + \
                "_" + str(self.fake_params["ampl_linear_norm_0"]) + ".fits"
            fits.writeto(filename = file_name_to_recon,
                         data = self.cube_frames,
                         overwrite = True)
            print("host_removal: "+str(datetime.datetime.now())+": Wrote cube which will be PCA-reconstructed as " + file_name_to_recon)
            print("-"*prog_bar_width)

            # the cube of PCA-reconstructed frames
            file_name_recon = self.config_data["data_dirs"]["DIR_OTHER_FITS"] + \
                injection_iteration_string + "/" + \
                "pca_recon_star_cube_" + \
                str(self.fake_params["angle_deg_EofN"]) + \
                "_" + str(self.fake_params["rad_asec"]) + \
                "_" + str(self.fake_params["ampl_linear_norm"]) + \
                "_" + str(self.fake_params["ampl_linear_norm_0"]) + ".fits"
            hdr1 = fits.Header()
            hdr1["ANGEOFN"] = self.fake_params["angle_deg_EofN"]
            hdr1["RADASEC"] = self.fake_params["rad_asec"]
            hdr1["AMPLIN"] = self.fake_params["ampl_linear_norm"]
            hdr1["AMPLIN0"] = self.fake_params["ampl_linear_norm_0"]
            fits.writeto(filename = file_name_recon,
                         data = recon_frames_cube_all_frames,
                         header = hdr1,
                         overwrite = True)
            print("host_removal: " + str(datetime.datetime.now()) + \
                ": Wrote PCA-reconstructed star cube to disk as " + file_name_recon)
            print("-"*prog_bar_width)

            # the cube of host-star-subtracted frames
            file_name = self.config_data["data_dirs"]["DIR_OTHER_FITS"] + \
                injection_iteration_string + "/" + \
                "host_removed_cube_" + \
                str(self.fake_params["angle_deg_EofN"]) + \
                "_" + str(self.fake_params["rad_asec"]) + \
                "_" + str(self.fake_params["ampl_linear_norm"]) + \
                "_" + str(self.fake_params["ampl_linear_norm_0"]) + ".fits"
            hdr = fits.Header()
            hdr["ANGEOFN"] = self.fake_params["angle_deg_EofN"]
            hdr["RADASEC"] = self.fake_params["rad_asec"]
            hdr["AMPLIN"] = self.fake_params["ampl_linear_norm"]
            hdr["AMPLIN0"] = self.fake_params["ampl_linear_norm_0"]
            fits.writeto(filename = file_name,
                         data = host_subt_cube_all_frames,
                         header = hdr,
                         overwrite = True)
            print("host_removal: " + str(datetime.datetime.now()) + \
                ": Wrote host-removed-cube to disk as " + file_name)
            print("-"*prog_bar_width)

        # for memory's sake
        del self.cube_frames

        print("host_removal: "+str(datetime.datetime.now())+" Returning cube of host-removed frames ")

        # return
        # host_subt_cube_all_frames: cube of non-derotated, host-star-subtracted frames
        # self.frame_num_array: array of the file name frame numbers (these are just passed without modification)
        return host_subt_cube_all_frames, self.frame_num_array


def main():
    '''
    Reconstruct and subtract host star PSFs from images
    '''

    # configuration data
    config = configparser.ConfigParser() # for parsing values in .init file
    config.read("/modules/config.ini")

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
    synthetic_data_host_removal_no_fake_planets = HostRemoval(n_PCA = 100,
                                                              outdir = config["data_dirs"]["DIR_FAKE_PSFS_HOST_REMOVED"], \
                                                              abs_PCA_name = str(config["data_dirs"]["DIR_PCA_CUBES_PSFS"] + \
                                                                                 "psf_PCA_vector_cookie_seqStart_000000_seqStop_010000.fits"),
                                                              abs_PCA_training_median = str(self.config_data["data_dirs"]["DIR_PCA_CUBES_PSFS"] + \
                                                                          'median_frame_seqStart_000000_seqStop_010000_pcaNum_0100.fits'))

    '''
    host_removal_fake_planets = HostRemoval(n_PCA = 100,
                                            outdir = config["data_dirs"]["DIR_FAKE_PSFS_HOST_REMOVED"], \
                                            abs_PCA_name = config["data_dirs"]["DIR_OTHER_FITS"] \
                                            + "pca_cubes_psfs/" \
                                            + "psf_PCA_vector_cookie_seqStart_004259_seqStop_005600.fits")

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
    '''
    pool.map(host_removal_fake_planets, fake_planet_frames_07_name_array)

    # remove the host from the frames WITHOUT fake planets
    ## ## host_removal_no_fake_planets(cookies_centered_06_directory[0])
    pool.map(host_removal_no_fake_planets_A, sci_frames_for_cube_A)
    pool.map(host_removal_no_fake_planets_B, sci_frames_for_cube_B)
    pool.map(host_removal_no_fake_planets_C, sci_frames_for_cube_C)
    pool.map(host_removal_no_fake_planets_D, sci_frames_for_cube_D)
    '''
