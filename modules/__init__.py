'''
Initialization
'''

import os
import numpy as np
import scipy
from scipy import ndimage, sqrt, stats, misc, signal
import git
import configparser
import multiprocessing
from sklearn.decomposition import PCA


## SOME VARIABLES
# number of CPUs for parallelization
ncpu = multiprocessing.cpu_count()

# configuration data
config = configparser.ConfigParser() # for parsing values in .init file
config.read("modules/config.ini")

# arcsec per radian
global asec_per_rad
asec_per_rad = np.divide(3600.*180.,np.pi)


## FUNCTIONS

def get_git_hash():
    '''
    Returns the hash for the current version of the code on Git
    '''

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    ## ## HAVENT FINISHED THIS YET
    print(sha)

    
def make_dirs():
    '''
    Make directories for housing files/info if they don't already exist
    '''

    # loop over all directory paths we will need
    for vals in config["data_dirs"]:
        abs_path_name = str(config["data_dirs"][vals])
        print("Directory exists: " + abs_path_name)
        
        # if directory does not exist, create it
        if not os.path.exists(abs_path_name):
            os.makedirs(abs_path_name)
            print("Made directory " + abs_path_name)
        

def make_first_pass_mask(quadChoice):
    '''
    Make mask for weird regions of the detector where I don't care about the background subtraction
    (This comes in when generating PCA basis of the background)

    RETURNS:
    mask_weird: array of nans and 1s for multiplying with the array to be masked
    '''
    
    mask_weird = np.ones((512,2048))
    mask_weird[0:10,:] = np.nan # edge
    mask_weird[-9:,:] = np.nan # edge
    mask_weird[:,0:10] = np.nan # edge
    mask_weird[260:,1046:1258] = np.nan # bullet hole
    mask_weird[:,1500:] = np.nan # unreliable bad pixel mask
    mask_weird[:,:440] = np.nan # unreliable bad pixel mask
    if quadChoice == 3: # if we want science on the third quadrant
        mask_weird[260:,:] = np.nan # get rid of whole top half
    if quadChoice == 2: # if we want science on the third quadrant
        mask_weird[:260,:] = np.nan # get rid of whole bottom half
    if np.logical_and(quadChoice!=2,quadChoice!=3):
        print('No detector science quadrant chosen!')
    
    return mask_weird


def find_airy_psf(image):
    '''
    Find star and return coordinates [y,x]
    (This comes in when generating PCA basis of the background)

    RETURNS:
    [cy, cx]: PSF center
    '''

    # replace NaNs with zeros to get the Gaussian filter to work
    nanMask = np.where(np.isnan(image) == True)
    image[nanMask] = 0
    
    # Gaussian filter for further removing effect
    # of bad pixels (somewhat redundant?)
    imageG = ndimage.filters.gaussian_filter(image, 6) 
    loc = np.argwhere(imageG==np.max(imageG))
    cx = loc[0,1]
    cy = loc[0,0]

    return [cy, cx]


def channels_PCA_cube():
    '''
    Generate PCA vector elements representing channel bias changes
    (This comes in when generating PCA basis of the background)

    RETURNS:
    channel_vars_PCA: cube of slices boolean-representing each channel
    '''

    # 32 channels, each 64 pixels wide
    total_channels = 32
    channel_vars_PCA = np.zeros((total_channels,512,2048))
    
    # in each slice, make one channel ones and leave the other pixels zero
    for chNum in range(0,total_channels):
        channel_vars_PCA[chNum,:,chNum*64:(chNum+1)*64] = 1.
        
    return channel_vars_PCA


def PCA_basis(training_cube_masked_weird, n_PCA):
    '''
    Return a PCA basis from a training cube

    INPUTS:
    training_cube_masked_weird: 3D cube of training images, with weird regions already masked
    (N.b. This cube needs to have dimensions [num_images,y_extent,x_extent])

    RETURNS:
    pca_basis: the PCA basis
    '''

    # get shape for a single image
    shape_img = np.shape(training_cube_masked_weird[0,:,:])
    
    # flatten each individual frame into a 1D array
    print("Flattening the training cube...")
    test_cube_1_1ds = np.reshape(training_cube_masked_weird,
                                     (np.shape(training_cube_masked_weird)[0],
                                      np.shape(training_cube_masked_weird)[1]*np.shape(training_cube_masked_weird)[2])
                                      ) 

    ## carefully remove nans before doing PCA

    # indices of finite elements over a single flattened frame
    idx = np.isfinite(test_cube_1_1ds[0,:])
        
    # reconstitute only the finite elements together in another PCA cube of 1D slices
    training_set_1ds_noNaN = np.nan*np.ones((len(test_cube_1_1ds[:,0]),np.sum(idx))) # initialize
        
    # for each PCA component, populate the arrays without nans with the finite elements
    for t in range(0,len(test_cube_1_1ds[:,0])): 
        training_set_1ds_noNaN[t,:] = test_cube_1_1ds[t,idx]

    # do PCA on the flattened `cube' with no NaNs
    print("Doing PCA to make PCA basis cube...")
    pca = PCA(n_components = n_PCA, svd_solver = "randomized") # initialize object
    #pca = RandomizedPCA(n_PCA) # for Python 2.7 
    test_pca = pca.fit(training_set_1ds_noNaN) # calculate PCA basis set
    del training_set_1ds_noNaN # clear memory

    ## reinsert the NaN values into each 1D slice of the PCA basis set
        
    print('Putting PCA components into cube...')

    # initialize a cube of 2D slices
    pca_comp_cube = np.nan*np.ones((n_PCA, shape_img[0], shape_img[1]), dtype = np.float32)

    # for each PCA component, populate the arrays without nans with the finite elements
    for slicenum in range(0,n_PCA):
        # initialize a new 1d frame long enough to contain all pixels
        pca_masked_1dslice_noNaN = np.nan*np.ones((len(test_cube_1_1ds[0,:])))
        # put the finite elements into the right positions
        pca_masked_1dslice_noNaN[idx] = pca.components_[slicenum]
        # put into the 2D cube
        pca_comp_cube[slicenum,:,:] = pca_masked_1dslice_noNaN.reshape(shape_img[0],shape_img[1]).astype(np.float32)

    return pca_comp_cube
