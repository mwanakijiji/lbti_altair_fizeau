
# coding: utf-8

# In[ ]:

# This does PCA background subtraction of the AC Her data


# In[1]:

from modules import *
import numpy as np
import matplotlib.pyplot as plt
import scipy
from astropy.io import fits
import pandas as pd
from datetime import datetime
import os
import sklearn
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
import time
get_ipython().magic(u'matplotlib inline')
#%matplotlib qt


# In[2]:

# stem 

stem = ('/home/../../media/unasemaje/Elements/lbti_data_reduction/180507_fizeau_altair/01_fix_pixed/second_attempt/')


# # FCN: MAKE PCA BASIS FROM TRAINING SET 
# # OF BACKGROUND FRAMES

# In[3]:

def make_pca_basis_cube_from_readouts(stem,startFrame,endFrame,quadChoice,indivChannel=False):
    '''
    INPUTS
    stem: filename stem for data
    startFrame: the first frame of the training set
    endFrame: the last frame of the training set
    quadchoice: the quadrant of the array we are interested in making a background for
    indivChannel: do you want to append PCA components that involve individual channel pedestals?
    
    OUTPUTS
    (none; FITS files are written out)
    '''

    # loop through each training set image and add it to a cube (endFrame is inclusive)
    print('Putting training frames into cube...')
    test_cube = np.nan*np.ones((endFrame-startFrame+1,511,2048), dtype = np.int64)
    
    mask_weird = make_first_pass_mask(quadChoice) # make the right mask
    
    for framenum in range(startFrame,endFrame+1): #endFrame+1): # 83, 67108
            
        # classically background-subtracted frames
        ##img_string = stem+'../02_background_subted/02a_subtraction_of_nod_off_median/'+'lm_180524_'+str("{:0>6d}".format(framenum))+'_02_02a.fits'

        # raw data (except that readout glitch correction has been done)
        img_string = stem+'lm_180507_'+str("{:0>6d}".format(framenum))+'.fits'
    
        # if FITS file exists in the first place
        if ((np.mod(framenum,1) == 0) & os.path.isfile(img_string)): 
            
            # read in image
            sciImg, header = fits.getdata(img_string,0,header=True)
        
            # mask weird parts of the readouts
            sciImg = np.multiply(sciImg,mask_weird)
        
            # add to cube
            test_cube[framenum-startFrame,:,:] = sciImg
            
        else:
            
            print('Hang on-- frame '+img_string+' not found!')
    
    # mask the raw training set
    test_cube = np.multiply(test_cube,mask_weird)
    
    # at this point, test_cube holds the (masked) background frames to be used as a training set
            
    # find the 2D median across all background arrays, and subtract it from each individual background array
    median_2d_bckgrd = np.nanmedian(test_cube, axis=0)    
    for t in range(0,endFrame-startFrame+1):
        test_cube[t,:,:] = np.subtract(test_cube[t,:,:],median_2d_bckgrd)         

    # at this point, test_cube holds the background frames which are dark- and 2D median-subtracted 

    # subtract the median from each individual background array (after it has been masked)
    for t in range(0,endFrame-startFrame+1):
        masked_slice = np.multiply(test_cube[t,:,:],mask_weird) # removes weird detector regions
        const_median = np.nanmedian(masked_slice)
        test_cube[t,:,:] = np.subtract(test_cube[t,:,:],const_median)   
        
    # at this point, test_cube holds the background frames which have had each frame's median value subtracted
        
    # remove a channel median from each channel in the science frame, since the  
    # channel pedestals will get taken out by the PCA components which are appended to the PCA
    # cube further below
    if indivChannel:
        print('Removing channel pedestals...')
        for slicenum in range(0,endFrame-startFrame+1): # loop over each slice
            for chNum in range(0,64): # loop over each channel in that slice
                test_cube[slicenum,:,chNum*64:(chNum+1)*64] = np.subtract(test_cube[slicenum,:,chNum*64:(chNum+1)*64],np.nanmedian(test_cube[slicenum,:,chNum*64:(chNum+1)*64]))
    
    # flatten each individual frame into a 1D array
    print('Flattening the training cube...')
    test_cube_1_1ds = np.reshape(test_cube,(np.shape(test_cube)[0],np.shape(test_cube)[1]*np.shape(test_cube)[2])) 
        
    ## carefully remove nans before doing PCA
    
    # indices of finite elements over a single flattened frame
    idx = np.isfinite(test_cube_1_1ds[0,:])
        
    # reconstitute only the finite elements together in another PCA cube of 1D slices
    training_set_1ds_noNaN = np.nan*np.ones((len(test_cube_1_1ds[:,0]),np.sum(idx))) # initialize array with slices the length of number of finite elements
    for t in range(0,len(test_cube_1_1ds[:,0])): # for each PCA component, populate the arrays without nans with the finite elements
        training_set_1ds_noNaN[t,:] = test_cube_1_1ds[t,idx]
    
    # do PCA on the flattened `cube' with no NaNs
    print('Doing PCA...')
    n_PCA = 12 # basis components
    #pca = PCA(n_components=n_PCA, svd_solver='randomized') # initialize object
    pca = RandomizedPCA(n_PCA) # for Python 2.7 
    test_pca = pca.fit(training_set_1ds_noNaN) # calculate PCA basis set
    del training_set_1ds_noNaN # clear memory
    
    # reinsert the NaN values into each 1D slice of the PCA basis set
    print('Putting PCA components into cube...')
    pca_comp_cube = np.nan*np.ones((n_PCA,511,2048), dtype = np.float32) # initialize a cube of 2D slices
    for slicenum in range(0,n_PCA): # for each PCA component, populate the arrays without nans with the finite elements
        pca_masked_1dslice_noNaN = np.nan*np.ones((len(test_cube_1_1ds[0,:]))) # initialize a new 1d frame long enough to contain all pixels
        pca_masked_1dslice_noNaN[idx] = pca.components_[slicenum] # put the finite elements into the right positions
        pca_comp_cube[slicenum,:,:] = pca_masked_1dslice_noNaN.reshape(511,2048).astype(np.float32) # put into the 2D cube

    # if I also want PCA slices for representing individual channel pedestal variations,
    # append slices representing each channel with ones
    extra_file_string = ''
    if indivChannel:
        
        # ... these slices that encode individual channel variations
        channels_alone = channels_PCA_cube()
        pca_comp_cube = np.concatenate((channels_alone,pca_comp_cube), axis=0)
        extra_file_string = '_w_channel_comps' # to add to filenames
    
    # save cube
    print('Saving PCA cube...')
    t = time.time()
    hdu = fits.PrimaryHDU(pca_comp_cube.astype(np.float32))
    del pca_comp_cube # clear memory
    hdul = fits.HDUList([hdu])
    
    
    hdul.writeto(stem+'/pca_cubes/background_PCA_hunzikerStyle_seqStart_'
                 +str("{:0>6d}".format(startFrame))+'_seqStop_'+str("{:0>6d}".format(endFrame))+extra_file_string+'.fits', 
                    overwrite=True)
    
    elapsed = time.time() - t
    print(elapsed)
    print('PCA cube saved as '+str("{:0>6d}".format(startFrame))+'_seqStop_'+str("{:0>6d}".format(endFrame))+extra_file_string+'.fits')
    print('---')
    


# In[ ]:

import ipdb; ipdb.set_trace()


# In[5]:

# these are in sections, because of memory limitations
make_pca_basis_cube_from_readouts(stem, 5200, 5299, 3, indivChannel=True)


# In[ ]:



