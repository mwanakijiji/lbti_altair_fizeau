'''
Prepares the data: bad pixel masking, background-subtraction, etc.
## ## This is descended from
## ## find_background_limit.ipynb
## ## subtract stray y illumination gradient parallel.py
## ## make_pca_basis_cube_altair.py
## ## pca_background_subtraction.ipynb
'''

import multiprocessing
import configparser
import glob
import time
from astropy.io import fits
from astropy.convolution import convolve, Gaussian1DKernel, interpolate_replace_nans
from regions import PixCoord, CircleSkyRegion, CirclePixelRegion, PolygonPixelRegion
from sklearn.decomposition import PCA
from modules import *

import matplotlib
matplotlib.use('agg') # avoids some crashes when multiprocessing
import matplotlib.pyplot as plt


class DarkSubtSingle:
    '''
    Dark-subtraction of a single raw frame
    '''

    def __init__(self, config_data=config):

        self.config_data = config_data

        # read in dark
        abs_dark_name = str(self.config_data["data_dirs"]["DIR_CALIB_FRAMES"]) + \
          "master_dark.fits"
        self.dark, self.header_dark = fits.getdata(abs_dark_name, 0, header=True)

    def __call__(self, abs_sci_name):
        '''
        Actual subtraction, for a single frame so as to parallelize job

        INPUTS:
        sci_name: science array filename
        '''

        # read in the science frame from raw data directory
        sci, header_sci = fits.getdata(abs_sci_name, 0, header=True)

        # subtract from image; data type should allow negative numbers
        image_dark_subtd = np.subtract(sci, self.dark).astype(np.int32)

        # add a line to the header indicating last reduction step
        header_sci["RED_STEP"] = "dark-subtraction"

        # write file out
        abs_image_dark_subtd_name = str(self.config_data["data_dirs"]["DIR_DARK_SUBTED"] + \
                                        os.path.basename(abs_sci_name))
        fits.writeto(filename=abs_image_dark_subtd_name,
                     data=image_dark_subtd,
                     header=header_sci,
                     overwrite=True)
        print("Writing out dark-subtracted frame " + os.path.basename(abs_sci_name))

        
class FixPixSingle:
    '''
    Interpolates over bad pixels
    '''

    def __init__(self, config_data=config):

        self.config_data = config_data

        # read in bad pixel mask
        # (altair 180507 bpm has convention 0=good, 1=bad)
        abs_badpixmask_name = str(self.config_data["data_dirs"]["DIR_CALIB_FRAMES"]) + \
          "master_bpm.fits"
        self.badpix, self.header_badpix = fits.getdata(abs_badpixmask_name, 0, header=True)

        # (altair 180507 bpm requires a top row)
        self.badpix = np.vstack([self.badpix,np.zeros(np.shape(self.badpix)[1])])

        # turn 1->nan (bad), 0->1 (good) for interpolate_replace_nans
        self.ersatz = np.nan*np.ones(np.shape(self.badpix))
        self.ersatz[self.badpix == 0] = 1.
        self.badpix = self.ersatz # rename
        del self.ersatz

        # define the convolution kernel (normalized by default)
        self.kernel = np.ones((3,3)) # just a patch around the kernel

    def __call__(self, abs_sci_name):
        '''
        Bad pix fixing, for a single frame so as to parallelize job

        INPUTS:
        sci_name: science array filename
        '''

        # read in the science frame from raw data directory
        sci, header_sci = fits.getdata(abs_sci_name, 0, header=True)

        # fix bad pixels (note conversion to 32-bit signed ints)
        sci_badnan = np.multiply(sci,self.badpix)
        image_fixpixed = interpolate_replace_nans(array=sci_badnan, kernel=self.kernel).astype(np.int32)

        # add a line to the header indicating last reduction step
        header_sci["RED_STEP"] = "bad-pixel-fixed"

        # write file out
        abs_image_fixpixed_name = str(self.config_data["data_dirs"]["DIR_PIXL_CORRTD"] + \
                                        os.path.basename(abs_sci_name))
        fits.writeto(filename=abs_image_fixpixed_name,
                     data=image_fixpixed,
                     header=header_sci,
                     overwrite=True)
        print("Writing out bad-pixel-fixed frame " + os.path.basename(abs_image_fixpixed_name))

    
class RemoveStrayRamp:
    '''
    Removes an additive electronic artifact illumination ramp in y at the top of the 
    LMIRcam readouts. (The ramp has something to do with resetting of the detector while 
    using the 2018A-era electronics; i.e., before MACIE overhaul in summer 2018-- see 
    emails from J. Leisenring and J. Stone, Sept. 5/6 2018)
    '''

    def __init__(self, config_data=config):
        self.config_data = config_data

    def __call__(self, abs_sci_name):
        '''
        Remove stray ramp, for a single frame so as to parallelize job

        INPUTS:
        sci_name: science array filename
        '''

        # read in the science frame from raw data directory
        sci, header_sci = fits.getdata(abs_sci_name, 0, header=True)

        # initialize kernel (for smoothing in y)
        num_pix_y = 5
        gauss_kernel = Gaussian1DKernel(num_pix_y)

        # find the median in x across the whole array as a function of y
        # (N.b. offsets in the channels shouldn't be a problem, since both
        # the ramp and the channel offsets should be additive)
        stray_ramp = np.nanmedian(sci[:,:],axis=1)

        # smooth it
        smoothed_stray_ramp = convolve(stray_ramp, gauss_kernel)

        # subtract from the whole array, after tiling across all pixels in x
        # (N.b., again, there will still be residual channel pedestals. These
        # are a mix of sky background and detector bias.)
        image_ramp_subted = np.subtract(sci,
                                        np.tile(smoothed_stray_ramp,(np.shape(sci)[1],1)).T).astype(np.float32)

        # add a line to the header indicating last reduction step
        header_sci["RED_STEP"] = "ramp_removed"

        # write file out
        abs_image_ramp_subted_name = str(self.config_data["data_dirs"]["DIR_RAMP_REMOVD"] + \
                                        os.path.basename(abs_sci_name))
        fits.writeto(filename=abs_image_ramp_subted_name,
                     data=image_ramp_subted,
                     header=header_sci,
                     overwrite=True)
        print("Writing out ramp-removed-fixed frame " + os.path.basename(abs_image_ramp_subted_name))

        
class BackgroundPCACubeMaker:
    '''
    Generates a PCA cube based on the backgrounds in the science frames.
    N.b. The user has to input a beginning and ending frame number for
    calculating a PCA basis (need to have common filter, integ time, etc.)
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
                 stop_frame_num,
                 quad_choice,
                 indiv_channel=False):
        '''
        Make PCA cube to reconstruct background (PSF is masked)

        INPUTS:
        start_frame_num: starting frame number to use in PCA basis generation
        stop_frame_num: stopping [ditto], inclusive
        quad_choice: quadrant (1 to 4) of the array we are interested in making a background for
        -> N.b. the quadrant we're making the background with should NOT contain a PSF; when
        -> the background is reconstructed for a science frame, the background WITHOUT a psf
        -> will be generated for a science fram quadrant that DOES contain a psf
        indivChannel: do you want to append PCA components that involve individual channel pedestals?
        '''

        # read in a first file to get the shape
        test_img, header = fits.getdata(self.file_list[0], 0, header=True)
        shape_img = np.shape(test_img)
        
        print("Initializing a PCA cube...")
        training_cube = np.nan*np.ones((stop_frame_num-start_frame_num+1,shape_img[0],shape_img[1]), dtype = np.int64)

        mask_weird = make_first_pass_mask(quad_choice) # make the right mask

        # initialize slice counter for removing unused slices later
        slice_counter = 0

        # loop over frames and add them to cube
        for frame_num in range(start_frame_num, stop_frame_num+1):

            # get name of file that this number corresponds to
            abs_matching_file_array = [s for s in self.file_list if str("{:0>6d}".format(frame_num)) in s]
            abs_matching_file = abs_matching_file_array[0] # get the name
            
            # if there was a match
            if (len(abs_matching_file) != 0):

                # read in the science frame from raw data directory
                sci, header_sci = fits.getdata(abs_matching_file, 0, header=True)

                # add to cube
                training_cube[slice_counter,:,:] = sci

                # advance counter
                slice_counter += 1

            # if there was no match
            elif (len(abs_matching_file) == 0):

                print("Frame " + os.path.basename(abs_matching_file) + " not found.")

            # if there were multiple matches
            else:

                print("Something is amiss with your frame number choice.")
                break

        # remove the unused slices
        training_cube = training_cube[0:slice_counter,:,:]

        # mask the raw training set
        training_cube_masked_weird = np.multiply(training_cube,mask_weird)
        del training_cube

        ## at this point, test_cube holds the (masked) background frames to be used as a training set

        # find the 2D median across all background arrays, and subtract it from each individual background array
        # (N.b. the region of the PSF itself will look funny, but we'll mask this in due course)
        median_2d_bckgrd = np.nanmedian(training_cube_masked_weird, axis=0)    
        for t in range(0,stop_frame_num-start_frame_num+1):
            training_cube_masked_weird[t,:,:] = np.subtract(training_cube_masked_weird[t,:,:],median_2d_bckgrd)

        ## at this point, test_cube holds the background frames which are dark- and 2D median-subtracted

        # remove a channel median from each channel in the science frame, since the
        # channel pedestals will get taken out by the PCA components which are appended to the
        # PCA cube further below
        if indiv_channel:
            print("Removing channel pedestals...")

            # loop over each frame in the cube
            for slice_num in range(0,stop_frame_num-start_frame_num+1):

                # loop over each channel in that frame (assumes 32 channels across, each 64 pixels wide)
                for ch_num in range(0,32): 
                    training_cube_masked_weird[slice_num,:,ch_num*64:(ch_num+1)*64] = \
                      np.subtract(training_cube_masked_weird[slice_num,:,ch_num*64:(ch_num+1)*64],
                                  np.nanmedian(training_cube_masked_weird[slice_num,:,ch_num*64:(ch_num+1)*64]))

        # generate the PCA cube from the empirical background data
        pca_comp_cube = PCA_basis(training_cube_masked_weird, n_PCA = self.n_PCA)
            
        # if we also want PCA slices for representing individual channel pedestal variations,
        # append slices representing each channel with ones
        if indiv_channel:
            # ... these slices that encode individual channel variations
            channels_alone = channels_PCA_cube()
            pca_comp_cube = np.concatenate((channels_alone,pca_comp_cube), axis=0)
                    
        # write out the PCA vector cube
        abs_pca_cube_name = str(self.config_data["data_dirs"]["DIR_OTHER_FITS"] +
                                'background_PCA_vector_quadrant_'+
                                str("{:0>2d}".format(quad_choice))+
                                '_seqStart_'+str("{:0>6d}".format(start_frame_num))+
                                '_seqStop_'+str("{:0>6d}".format(stop_frame_num))+'.fits')
        fits.writeto(filename=abs_pca_cube_name,
                     data=pca_comp_cube,
                     header=None,
                     overwrite=True)
        print("Wrote out background PCA cube " + os.path.basename(abs_pca_cube_name))
        
        
class BackgroundPCASubtSingle:
    '''
    Does a PCA decomposition of the background of a given sequence of science frames,
    and subtracts it from each frame (N.b. remaining pedestal should be photons alone)
    
    Steps are
    1. reads in PCA component cube
    2. masks and subtracts the median (just a constant) from each science frame
    3. decomposes each science frame into its PCA components (with a mask over the PSF)
    4. subtracts the reconstructed background
    5. saves the background-subtracted images
    '''
    
    def __init__(self, inputArray, config_data=config):
        '''
        INPUTS:

        inputArray: a 1D array with 
        -> [0]: cube_start_framenum: starting frame number of the PCA component cube
        -> [1]: cube_stop_framenum: stopping frame number (inclusive)  "  "
        -> [2]: n_PCA: number of PCA components to reconstruct the background with,
        --------> out of the total number of components in the cube
        -> [3]: background quadrant choice (2 or 3)
        config_data: configuration data, as usual
        '''
        
        self.config_data = config_data

        # unpack values
        self.cube_start_framenum = inputArray[0]
        self.cube_stop_framenum = inputArray[1]
        self.n_PCA = inputArray[2]
        self.quad_choice = inputArray[3]

        # read in PCA cube
        cube_string = str(self.config_data["data_dirs"]["DIR_OTHER_FITS"] +
                                'background_PCA_vector_quadrant_'+
                                str("{:0>2d}".format(self.quad_choice))+
                                '_seqStart_'+str("{:0>6d}".format(self.cube_start_framenum))+
                                '_seqStop_'+str("{:0>6d}".format(self.cube_stop_framenum))+'.fits')
        self.pca_cube = fits.getdata(cube_string,0,header=False)#.astype(np.float32)

        # apply mask over weird detector regions to PCA cube
        # (N.b. otherwise the PCA wont converge!)
        self.pca_cube = np.multiply(self.pca_cube,make_first_pass_mask(self.quad_choice))
    
    def __call__(self, abs_sci_name):
        '''
        PCA-based background subtraction, for a single frame so as to parallelize job

        INPUTS:
        abs_sci_name: science array filename
        '''

        # start the timer
        start_time = time.time()
        
        # read in the science frame from raw data directory
        sciImg, header_sci = fits.getdata(abs_sci_name, 0, header=True)

        # apply mask over weird detector regions to science image
        sciImg = np.multiply(sciImg, make_first_pass_mask(self.quad_choice))
        
        
        ## mask the PSF
        
        # define region
        psf_loc = find_airy_psf(sciImg) # center of science PSF
        print("PSF location in " + os.path.basename(abs_sci_name) + ": [" + str(psf_loc[0]) + ", " + str(psf_loc[1]) + "]")
        radius = 30. # radius around PSF that will be masked
        center = PixCoord(x=psf_loc[1], y=psf_loc[0])
        region = CirclePixelRegion(center, radius)
        mask_psf_region = region.to_mask()

        # apply the mask to science array
        psf_mask = np.ones(np.shape(sciImg)) # initialize arrays of same size as science image
        mask_psf_region.data[mask_psf_region.data == 1] = np.nan  # make zeros within mask cutout (but not in the mask itself) nans
        mask_psf_region.data[mask_psf_region.data == 0] = 1
        ##mask_psf_region.data[mask_psf_region.data == -99999] = 0 # have to avoid nans in the linear algebra
        psf_mask[mask_psf_region.bbox.slices] = mask_psf_region.data  # place the mask cutout (consisting only of 1s) onto the array of nans

        # I don't know why, but the sciImg nans in weird detector regions become zeros by this point, and I want them to stay nans
        # so, re-apply the mask over the bad regions of the detector
        sciImg = np.multiply(sciImg, make_first_pass_mask(self.quad_choice))
        
        sciImg_masked = np.multiply(sciImg,psf_mask) # this is now the masked science frame  
        
        # subtract the median (just a constant) from the remaining science image
        sciImg_psf_masked = np.subtract(sciImg_masked,np.nanmedian(sciImg_masked)) # where PSF is masked
        sciImg_psf_not_masked = np.subtract(sciImg,np.nanmedian(sciImg_masked)) # where PSF is not masked
        
        # apply the PSF mask to PCA slices, with which we will do the fitting
        pca_cube_masked = np.multiply(self.pca_cube,psf_mask)


        ## PCA-decompose
        
        # flatten the science array and PCA cube 
        pca_not_masked_1ds = np.reshape(self.pca_cube,(np.shape(self.pca_cube)[0],np.shape(self.pca_cube)[1]*np.shape(self.pca_cube)[2]))
        sci_masked_1d = np.reshape(sciImg_psf_masked,(np.shape(sciImg_masked)[0]*np.shape(sciImg_masked)[1]))
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
        soln_vector = np.linalg.lstsq(pca_masked_1ds_noNaN[0:self.n_PCA,:].T, sci_masked_1d_noNaN)
        
        # reconstruct the background based on that vector
        # note that the PCA components WITHOUT masking of the PSF location is being
        # used to reconstruct the background
        recon_backgrnd_2d = np.dot(self.pca_cube[0:self.n_PCA,:,:].T, soln_vector[0]).T
        # now do the same, but for the channel bias variation contributions only (assumes 32 elements only)
        recon_backgrnd_2d_channels_only_no_psf_masking = np.dot(self.pca_cube[0:32,:,:].T, soln_vector[0][0:32]).T # without PSF masking
        recon_backgrnd_2d_channels_only_psf_masked = np.dot(self.pca_cube[0:32,:,:].T, soln_vector[0][0:32]).T # with PSF masking
        
        # do the actual subtraction:
        # all-background subtraction
        sciImg_subtracted = np.subtract(sciImg_psf_not_masked,recon_backgrnd_2d)
        # background subtraction of channels only:
        # without PSF masking
        #sciImg_subtracted_channels_only_no_psf_masking = np.subtract(sciImg_psf_not_masked,recon_backgrnd_2d_channels_only)
        # with PSF masking
        sciImg_subtracted_channels_only_psf_masked = np.subtract(sciImg_psf_masked,
                                                                 np.multiply(recon_backgrnd_2d_channels_only_psf_masked,np.multiply(self.pca_cube,psf_mask))) 

        # add last reduction step to header
        header_sci["RED_STEP"] = "pca_background_subtracted"
        
        # save reconstructed background for checking
        abs_recon_bkgd = str(self.config_data["data_dirs"]["DIR_OTHER_FITS"] +
                                'recon_bkgd_quad_'+
                                str("{:0>2d}".format(self.quad_choice))+
                                '_PCAseqStart_'+str("{:0>6d}".format(self.cube_start_framenum))+
                                '_PCAseqStop_'+str("{:0>6d}".format(self.cube_stop_framenum))+'_'+
                                os.path.basename(abs_sci_name))
        fits.writeto(filename=abs_recon_bkgd,
                     data=recon_backgrnd_2d,
                     overwrite=True)
        
        # save masked science frame BEFORE background-subtraction
        abs_masked_sci_before_bkd_subt = str(self.config_data["data_dirs"]["DIR_OTHER_FITS"] +
                                'masked_sci_before_bkd_subt_quad_'+
                                str("{:0>2d}".format(self.quad_choice))+
                                '_PCAseqStart_'+str("{:0>6d}".format(self.cube_start_framenum))+
                                '_PCAseqStop_'+str("{:0>6d}".format(self.cube_stop_framenum))+
                                             os.path.basename(abs_sci_name))
        fits.writeto(filename=abs_masked_sci_before_bkd_subt,
                     data=sciImg_psf_masked,
                     overwrite=True)

        # save masked, background-subtracted science frame
        background_subtracted_masked = np.multiply(sciImg_subtracted,make_first_pass_mask(self.quad_choice))
        background_subtracted_masked = np.multiply(background_subtracted_masked,psf_mask)
        abs_masked_sci_after_bkd_subt = str(self.config_data["data_dirs"]["DIR_OTHER_FITS"] +
                                'masked_sci_after_bkd_subt_quad_'+
                                str("{:0>2d}".format(self.quad_choice))+
                                '_PCAseqStart_'+str("{:0>6d}".format(self.cube_start_framenum))+
                                '_PCAseqStop_'+str("{:0>6d}".format(self.cube_stop_framenum))+
                                             os.path.basename(abs_sci_name))
        fits.writeto(filename=abs_masked_sci_after_bkd_subt,
                     data=background_subtracted_masked,
                     overwrite=True)
            
        
        # save background-subtracted science frame
        sciImg_subtracted_name = str(self.config_data["data_dirs"]["DIR_PCAB_SUBTED"] +
                                os.path.basename(abs_sci_name))
        fits.writeto(filename=sciImg_subtracted_name,
                     data=sciImg_subtracted,
                     header=header_sci,
                     overwrite=True)
        print('Background-subtracted frame '+
              os.path.basename(abs_sci_name)+
              ' written out. PCA = '+str(self.n_PCA))

        ## make FYI plots of the effectiveness of the background subtraction
        # (N.b. the PSF and weird detector regions are masked here)
        self.vital_stats(file_base_name = str(os.path.basename(abs_sci_name)),
                         sci_img_pre = sciImg_masked,
                         sci_img_post_channel_subt = sciImg_subtracted_channels_only_psf_masked,
                         sci_img_post_all_subt = background_subtracted_masked,
                         pca_spec = soln_vector[0])
        
        print('Elapsed time:')
        elapsed_time = time.time() - start_time
        print('--------------------------------------------------------------')
        print(elapsed_time)
        

    def return_array_one_block(self, sliceArray):
        '''
        This takes a 1D array with background frame range, science frame range, and N_PCA information
        and returns an expanded array where each row corresponds to a single science array
        '''
    
        # INPUT: an array containing 
        # [0]: starting frame of background sequence
        # [1]: ending frame of background sequence (inclusive)
        # [2]: starting science frame to background-subtract
        # [3]: ending science frame to background-subtract (inclusive)
        # [4]: number of PCA components to use in background reconstruction
        # N.b. There already needs to be a PCA background cube corresponding to [0] and [1]
    
        # OUTPUT: an array of arrays where each element corresponds to the 
        # parameters of a single science image (i.e., the input array elements
        # [0], [1], [4] are replicated for each science frame. 
    
        # unpack some values
        science_start_frame = sliceArray[2]
        science_end_frame = sliceArray[3]
    
        sliceArrayTiled = np.tile(sliceArray,(science_end_frame-science_start_frame+1,1)) # tile, where each row corresponds to a science frame
        sliceArrayTiled2 = np.delete(sliceArrayTiled,2,1) # delete col [2]

        # convert new col [2] (old col [3]) to be entries for individual frame numbers
        for sciframeNum in range(science_start_frame,science_end_frame+1):
            t = int(sciframeNum-science_start_frame) # index denoting the row
            sliceArrayTiled2[t][2] = int(sciframeNum) # insert frame number
    
        # The table now involves columns
        # [0]: background_start_frame
        # [1]: background_end_frame
        # [2]: science frame number
        # [3]: number of PCA components to reconstruct the background

        return sliceArrayTiled2


    def vital_stats(self, file_base_name, sci_img_pre, sci_img_post_channel_subt, sci_img_post_all_subt, pca_spec):
        '''
        Make a FYI plot of the PCA spectrum, and the noise levels before and after subtraction

        INPUTS:
        file_base_name: file base name (in the form of an os.path.basename)
        sci_img_pre: science array, before background-subtraction
        sci_img_post_channel_subt: science array, after subtraction of channel variations only
        sci_img_post_all_subt: science array, after subtraction of the whole reconstructed background
        pca_spec: the PCA spectrum, including the channel variations
        -> N.b. for now it is assumed that the first 32 elements are the channel variations

        RETURNS:
        Saves plot of 5 subplots:
        0. Histogram of counts before any background-subtraction
        1. PCA spectrum of the background decomposition (channel bias variations only)
        2. PCA spectrum of the background decomposition (noise from photons, read, etc.; no channel bias variations)
        3. Histogram of counts after channel bias subtraction
        4. Histogram of counts after channel bias and other noise subtraction
        '''
    
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,10))
        
        # Histogram of counts before any background-subtraction
        array_ravelPre = np.ravel(sci_img_pre)
        iiPre = np.isfinite(array_ravelPre)
        axes[0,0].set_title('Pre-backgd subtraction\nMedian: '+str(np.nanmedian(sci_img_pre))+'\nStdev: '+str(np.nanstd(sci_img_pre)))
        axes[0,0].hist(array_ravelPre[iiPre], bins=200)
        axes[0,0].set_xlim([-1000,1000])

        # PCA spectrum of the background decomposition (channel bias variations only)
        axes[0,1].set_title('PCA Spec\n(channels)')
        axes[0,1].plot(pca_spec)
        axes[0,1].set_xlim([0,32])

        # PCA spectrum of the background decomposition (noise from photons, read, etc.; no channel bias variations)
        axes[0,2].set_title('PCA Spec\n(not incl. channels)')
        axes[0,2].plot(pca_spec)
        axes[0,2].set_xlim([32,len(pca_spec)])
        axes[0,2].set_ylim([np.min(pca_spec[32:]),np.max(pca_spec[32:])])

        # Histogram of counts after subtraction of background contributed by channel bias variations
        array_ravelPost_channel_subt = np.ravel(sci_img_post_channel_subt)
        iiiPost = np.isfinite(array_ravelPost_channel_subt)
        axes[1,0].set_title('Post-channel background subtraction\nMedian: '+
                          str(np.nanmedian(array_ravelPost_channel_subt))+
                          '\nStdev: '+str(np.nanstd(array_ravelPost_channel_subt)))
        axes[1,0].hist(array_ravelPost_channel_subt[iiiPost], bins=200)
        axes[1,0].set_xlim([-1000,1000])

        print('----------------------')
        # some tests of regions where the background is not being oversubtracted due to the PSF
        print('before any subt, '+file_base_name+': std='+str(np.nanstd(sci_img_pre[:,0:760]))+', med='+str(np.nanmedian(sci_img_pre[:,0:760])))
        print('after channels only, '+file_base_name+': std='+str(np.nanstd(sci_img_post_channel_subt[:,0:760]))+', med='+str(np.nanmedian(sci_img_post_channel_subt[:,0:760])))
        print('after all, '+file_base_name+': std='+str(np.nanstd(sci_img_post_all_subt[:,0:760]))+', med='+str(np.nanmedian(sci_img_post_all_subt[:,0:760])))

        # Histogram of counts after BOTH channel bias and other noise subtraction
        array_ravelPost_all_subt = np.ravel(sci_img_post_all_subt)
        ivPost = np.isfinite(array_ravelPost_all_subt)
        axes[1,1].set_title('Post-all background subtraction\nMedian: '+
                          str(np.nanmedian(array_ravelPost_all_subt))+'\nStdev: '+
                          str(np.nanstd(array_ravelPost_all_subt)))
        axes[1,1].hist(array_ravelPost_all_subt[ivPost], bins=200)
        axes[1,1].set_xlim([-1000,1000])

        plt.suptitle("frame " + file_base_name )
        plt.tight_layout()
        abs_vital_stats_png = str(self.config_data["data_dirs"]["DIR_FYI_INFO"] +
                                '/bkgd_subt_stats_' + file_base_name.split(".")[0] + '.png')
        plt.savefig(abs_vital_stats_png)
        plt.clf()

    
class CalcNoise:
    '''
    Finds noise characteristics: where is the background limit? etc.
    '''

    pass


class CookieCutout:
    '''
    Cuts out region around PSF commensurate with the AO control radius
    '''

    def __init__(self, quad_choice, config_data=config):

        self.config_data = config_data

        # determine approximate AO control radius r_c in detector pixels
        # r_c = lambda/(2*d)
        d = np.divide(np.float(self.config_data["instrum_params"]["D_1"]),
                      np.float(self.config_data["instrum_params"]["N_SUBAP_DIAM"])) # intersubaperture spacing (m)
        r_c = np.divide(np.float(self.config_data["observ_params"]["WAVEL_C_UM"]),2*d)*(10**-6)*asec_per_rad # r_c (m)
        self.ao_ctrl_pix = np.divide(r_c,np.float(self.config_data["instrum_params"]["LMIR_PS"]))
        self.buffer_fac = 0.5 # multiple of r_c we want to cut out around the PSF
        ## ## (SMALL VALUE OF BUFFER IN PLACE FOR NOW, UNTIL I CHANGE THE METHOD OF BACKGROUND SUBTRACTION)
        self.quad_choice = quad_choice

    def __call__(self, abs_sci_name):
        '''
        Cut out the cookie around the PSF

        INPUTS:
        abs_sci_name: absolute science array filename
        quad_choice: quadrant in which this PSF sits (2 or 3)
        '''

        print("AO control radius in pixels: " + str(int(self.ao_ctrl_pix)))

        # read in the science frame from raw data directory
        sciImg, header_sci = fits.getdata(abs_sci_name, 0, header=True)

        # apply mask over weird detector regions to science image
        sciImg = np.multiply(sciImg, make_first_pass_mask(self.quad_choice))
        
        # with crude centering, cut out squares around that region
        # (we'll center more carefully next)
        psf_loc = find_airy_psf(sciImg)

        cookie_cut_out = sciImg[psf_loc[0]-int(self.buffer_fac*self.ao_ctrl_pix):psf_loc[0]+int(self.buffer_fac*self.ao_ctrl_pix),
                                psf_loc[1]-int(self.buffer_fac*self.ao_ctrl_pix):psf_loc[1]+int(self.buffer_fac*self.ao_ctrl_pix)]

        # add a line to the header indicating last reduction step
        header_sci["RED_STEP"] = "cookie_cutout"

        # write out
        print("Writing out squares of half-width " + str(self.buffer_fac)+\
              " of the control radius for " + os.path.basename(abs_sci_name))
        abs_cookie_subarray = str(self.config_data["data_dirs"]["DIR_CUTOUTS"] + \
                                  os.path.basename(abs_sci_name))
        fits.writeto(filename=abs_cookie_subarray,
                     data=cookie_cut_out,
                     header=header_sci,
                     overwrite=True)
        
    

def main():
    '''
    Carry out the basic data-preparation steps
    '''

    # configuration data
    config = configparser.ConfigParser() # for parsing values in .init file
    config.read("modules/config.ini")

    # multiprocessing instance
    pool = multiprocessing.Pool(ncpu)

    # make a list of the raw files
    raw_00_directory = str(config["data_dirs"]["DIR_RAW_DATA"])
    raw_00_name_array = list(glob.glob(os.path.join(raw_00_directory, "*.fits")))

    # subtract darks in parallel
    print("Subtracting darks with " + str(ncpu) + " CPUs...")
    do_dark_subt = DarkSubtSingle(config)
    pool.map(do_dark_subt, raw_00_name_array)

    # make a list of the dark-subtracted files
    darksubt_01_directory = str(config["data_dirs"]["DIR_DARK_SUBTED"])
    darksubt_01_name_array = list(glob.glob(os.path.join(darksubt_01_directory, "*.fits")))

    # fix bad pixels in parallel
    print("Fixing bad pixels with " + str(ncpu) + " CPUs...")
    do_fixpix = FixPixSingle(config)
    pool.map(do_fixpix, darksubt_01_name_array)

    # make a list of the bad-pix-fixed files
    fixpixed_02_directory = str(config["data_dirs"]["DIR_PIXL_CORRTD"])
    fixpixed_02_name_array = list(glob.glob(os.path.join(fixpixed_02_directory, "*.fits")))

    # subtract ramps in parallel
    print("Subtracting artifact ramps with " + str(ncpu) + " CPUs...")
    do_ramp_subt = RemoveStrayRamp(config)
    pool.map(do_ramp_subt, fixpixed_02_name_array)
    
    # make a list of the ramp-removed files
    ramp_subted_03_directory = str(config["data_dirs"]["DIR_RAMP_REMOVD"])
    ramp_subted_03_name_array = list(glob.glob(os.path.join(ramp_subted_03_directory, "*.fits")))

    # generate PCA cubes for backgrounds
    # (N.b. n_PCA needs to be smaller that the number of frames being used, and
    # this number does NOT include possible elements representing individual
    # channel bias variations)
    pca_backg_maker = BackgroundPCACubeMaker(file_list = ramp_subted_03_name_array,
                                            n_PCA = 10) # create instance
    pca_backg_maker(start_frame_num = 9000,
                   stop_frame_num = 9099,
                   quad_choice = 2,
                   indiv_channel = True)

    '''                  
    # PCA-based background subtraction in parallel
    print("Subtracting backgrounds with " + str(ncpu) + " CPUs...")
    
    # set up parameters of PCA background-subtraction:
    # [0]: starting frame number of the PCA component cube
    # [1]: stopping frame number (inclusive)  "  "
    # [2]: total number of PCA components to reconstruct the background with
    # [3]: background quadrant choice (2 or 3)
    ## WILL NEED MORE PARAMETER ARRAYS FOR VARIOUS FRAME SEQUENCES
    param_array = [9000, 9099, 42, 2]
    do_pca_back_subt = BackgroundPCASubtSingle(param_array, config)
    pool.map(do_pca_back_subt, ramp_subted_03_name_array)
    '''
    
    # make a list of the PCA-background-subtracted files
    pcab_subted_04_directory = str(config["data_dirs"]["DIR_PCAB_SUBTED"])
    pcab_subted_04_name_array = list(glob.glob(os.path.join(pcab_subted_04_directory, "*.fits")))

    # make cookie cutouts of the PSFs
    ## ## might add functionality to override the found 'center' of the PSF
    make_cookie_cuts = CookieCutout(quad_choice = 2)
    pool.map(make_cookie_cuts, pcab_subted_04_name_array)    
