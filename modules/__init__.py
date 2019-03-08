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
        

# mask for weird regions of the detector where I don't care about the background subtraction
# (This comes in when generating PCA basis of the background)
def make_first_pass_mask(quadChoice):
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


# find star and return coordinates [y,x]
# (This comes in when generating PCA basis of the background)
def find_airy_psf(image):

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


# generate PCA vector elements representing channel bias changes
# (This comes in when generating PCA basis of the background)
def channels_PCA_cube():

    # 32 channels, each 64 pixels wide
    total_channels = 32
    channel_vars_PCA = np.zeros((total_channels,512,2048))
    
    # in each slice, make one channel ones and leave the other pixels zero
    for chNum in range(0,total_channels):
        channel_vars_PCA[chNum,:,chNum*64:(chNum+1)*64] = 1.
        
    return channel_vars_PCA
