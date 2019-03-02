'''
Initialization
'''

import os
import numpy as np
import scipy
from scipy import ndimage, sqrt, stats, misc, signal
import git
import configparser

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

    config = configparser.ConfigParser() # for parsing values in .init file
    config.read("modules/altair_config.ini")

    # loop over all directory paths we will need
    for path_num in range(0,len(config["data_dirs"])):

        # check/make a needed directory, skipping the stem itself
        os.makedirs(config["data_dirs"]["STEM"] + config["data_dirs"][1+path_num])

'''
# mask for weird regions of the detector where I don't care about the background subtraction
def make_first_pass_mask(quadChoice):
    mask_weird = np.ones((511,2048))
    mask_weird[0:4,:] = np.nan # edge
    mask_weird[-4:,:] = np.nan # edge
    mask_weird[:,0:4] = np.nan # edge
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
def find_airy_psf(image):

    # replace NaNs with zeros to get the Gaussian filter to work
    nanMask = np.where(np.isnan(image) == True)
    image[nanMask] = 0
    
    # Gaussian filter
    imageG = ndimage.filters.gaussian_filter(image, 6) # further remove effect of bad pixels (somewhat redundant?)
    loc = np.argwhere(imageG==np.max(imageG))
    cx = loc[0,1]
    cy = loc[0,0]

    return [cy, cx]

def channels_PCA_cube():
    
    channel_vars_PCA = np.zeros((64,511,2048))
    
    # in each slice, make one channel ones and leave the other pixels zero
    for chNum in range(0,64):
        channel_vars_PCA[chNum,:,chNum*64:(chNum+1)*64] = 1.
        
    return channel_vars_PCA
'''
