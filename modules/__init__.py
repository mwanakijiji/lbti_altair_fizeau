'''
Initialization
'''

import os
import time
import numpy as np
import scipy
from scipy import ndimage, sqrt, stats, misc, signal
#import git
import configparser
import multiprocessing
from astropy.io import fits
from sklearn.decomposition import PCA


## SOME VARIABLES
# number of CPUs for parallelization
ncpu = multiprocessing.cpu_count()

# configuration data
#global config
config = configparser.ConfigParser() # for parsing values in .init file
config.read("./modules/config.ini")

# status/progress/parse bar length
prog_bar_width = 30

# arcsec per radian
global asec_per_rad
asec_per_rad = np.divide(3600.*180.,np.pi)

# PSF constants
lambda_over_D_pix = 9.46 # for Airy PSF, 4 um with LBT
fwhm_4um_lbt_airy_pix = 1.028*lambda_over_D_pix # fwhm of a 4 um Airy PSF with the LBT

# change in companion amplitude (ascending iteration)
# (in linear units normalized to star; note these are STEP sizes)
del_amplitude_progression = np.array([0.5e-3,1e-4,5e-5,1e-5,5e-6,1e-6,5e-7,1e-7,5e-8,1e-8])


## FUNCTIONS

'''
def get_git_hash():

    # Returns the hash for the current version of the code on Git

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    ## ## HAVENT FINISHED THIS YET
    print(sha)
'''


def polar_to_xy(pos_info, pa, asec = False, south = False, north = False):
    '''
    Converts polar vectors (deg, pix) to xy vectors (pix, pix)
    which incorporate the parallactic angle
    (Note degrees are CCW from +x axis)

    INPUTS:
    pos_info: dictionary with keys
        "rad_pix": radius in pixels (if asec = False)
        "rad_asec": radius in arcsec (if asec = True)
        "angle_deg_EofN": angle in degrees E of true N
    pa: parallactic angle (or if no rotation compensation
        desired, just use 0)
    asec: flag as to whether the radius is in asec
        (otherwise, it should be in pix)
    south: flag as to whether target is in south
    north: flag as to whether target is in south

    OUTPUTS:
    dictionary with the addition of keys
        "x_pix_coord": position in x in pixels
        "y_pix_coord": position in y in pixels
    '''
    # sanity check
    if (south and north):
        raw_input("Nonsensical flags: target is in both south and north!")

    # if radius is in asec
    if asec:
        pos_info["rad_pix"] = np.divide(pos_info["rad_asec"],
                                        np.float(config["instrum_params"]["LMIR_PS"]))

    # Convert to x,y

    # Consider variables
    # PA: parallactic angle
    # theta: angle E of true N (i.e., CCW from +y axis for a southern target, after derotation of image)
    # R: radial distance from origin (which is at the central host star)

    # In non-derotated frame, the coordinate of interest is
    # (x,y) = R*( sin(-PA-theta), cos(-PA-theta) )

    pos_info["x_pix_coord"] = np.multiply(pos_info["rad_pix"],
                                          np.sin(np.multiply(np.add(-pos_info["angle_deg_EofN"],-pa),np.pi/180.)))
    pos_info["y_pix_coord"] = np.multiply(pos_info["rad_pix"],
                                          np.cos(np.multiply(np.add(-pos_info["angle_deg_EofN"],-pa),np.pi/180.)))

    # if target is in north (but higher than Polaris), use the same relations except take -pa -> +pa
    if north:
        print("Calculating coordinates for a northern target...")
        pos_info["x_pix_coord"] = np.multiply(pos_info["rad_pix"],
                                          np.sin(np.multiply(np.add(-pos_info["angle_deg_EofN"],pa),np.pi/180.)))
        pos_info["y_pix_coord"] = np.multiply(pos_info["rad_pix"],
                                          np.cos(np.multiply(np.add(-pos_info["angle_deg_EofN"],pa),np.pi/180.)))

    return pos_info


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

def simple_save_fits(image, file_name):
    '''
    Handy function to save a FITS image locally for bug-checking
    '''
    fits.writeto(filename = file_name,data = image,overwrite = True)


def make_first_pass_mask(image, quadChoice):
    '''
    Make mask for weird regions of the detector where I don't care about the background subtraction
    (This comes in when generating PCA basis of the background)

    INPUTS:
    image: the 2D image to be masked
    quadChoice: the quadrant the PSF is in (2 or 3, for Altair dataset)

    RETURNS:
    mask_weird: array of nans and 1s for multiplying with the array to be masked
    '''

    # if image is 2D
    if (len(np.shape(image)) == 2):
        image[0:10,:] = np.nan
        image[-9:,:] = np.nan # edge
        image[:,0:10] = np.nan # edge
        image[260:,1046:1258] = np.nan # bullet hole
        image[:,1500:] = np.nan # unreliable bad pixel mask
        image[:,:440] = np.nan # unreliable bad pixel mask
        # The below was commented out to try subtracting only the channel variations and get more radius around star
        '''
        if quadChoice == 3: # if we want science on the third quadrant
            image[260:,:] = np.nan # get rid of whole top half
        if quadChoice == 2: # if we want science on the third quadrant
            image[:260,:] = np.nan # get rid of whole bottom half
        '''

    # if image is a 3D cube
    elif (len(np.shape(image)) == 3):
        image[:,0:10,:] = np.nan
        image[:,-9:,:] = np.nan # edge
        image[:,:,0:10] = np.nan # edge
        image[:,260:,1046:1258] = np.nan # bullet hole
        image[:,:,1500:] = np.nan # unreliable bad pixel mask
        image[:,:,:440] = np.nan # unreliable bad pixel mask
        # The below was commented out to try subtracting only the channel variations and get more radius around star
        '''
        if quadChoice == 3: # if we want science on the third quadrant
            image[:,260:,:] = np.nan # get rid of whole top half
        if quadChoice == 2: # if we want science on the third quadrant
            image[:,:260,:] = np.nan # get rid of whole bottom half
        '''

    # deal with a bug associated with 32-bit floats,
    # where stray pixels (in my experience, just one) can be assigned value +-2.1474842e+09
    # (left un-fixed, it screws up the whole PCA decomposition)
    image[np.abs(image) > 1e+05] = 0

    if np.logical_and(quadChoice!=2,quadChoice!=3):
        print('No detector science quadrant chosen! (But that may not be a problem if you are just ' + \
              'subtracting the channel variations.)')

    return image


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
    # add extra dimension for 2D array; vestigial
    '''
    if np.shape(training_cube_masked_weird) == 2:
        training_cube_masked_weird = np.expand_dims(training_cube_masked_weird, axis=0)
    '''
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


def fit_pca_star(pca_cube, sciImg, raw_pca_training_median, mask_weird, n_PCA, subt_median=True):
    '''
    INPUTS:
    pca_cube: cube of PCA components
    sciImg: the science image
    raw_pca_training_median: the median offset which needs to be subtracted
        from the science image before reconstructing the residuals,
        and then added back in after the PCA reconstruction
        (if none is needed, then just feed in an array of zeros)
    mask_weird: mask defining areas which are to be interpolated over
    n_PCA: number of PCA components
    subt_median: if True, subtract raw_pca_training_median and add it back in
        after the PCA stuff; if False, just PCA-decompose the sciImg as-is
        (and raw_pca_training_median is not used)

    RETURNS:
    pca_vector: spectrum of PCA vector amplitudes
    recon_2d: host star PSF as reconstructed with N PCA vector components
    recon_2d_masked: same as recon_2d, but with the 'weird_mask'
    '''

    # apply mask over weird regions to PCA cube
    try:
        pca_cube_masked = np.multiply(pca_cube,mask_weird)
    except:
        print("Mask and input image have incompatible dimensions!")
        return

    # apply mask over weird detector regions to science image
    sciImg_psf_masked = np.multiply(sciImg,mask_weird)

    if subt_median:
        # subtract the offset
        sciImg_psf_masked = np.subtract(sciImg_psf_masked,raw_pca_training_median)

    ## PCA-decompose

    # flatten the science array and PCA cube
    pca_not_masked_1ds = np.reshape(pca_cube,(np.shape(pca_cube)[0],np.shape(pca_cube)[1]*np.shape(pca_cube)[2]))
    sci_masked_1d = np.reshape(sciImg_psf_masked,(np.shape(sciImg_psf_masked)[0]*np.shape(sciImg_psf_masked)[1]))
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
    soln_vector = np.linalg.lstsq(pca_masked_1ds_noNaN[0:n_PCA,:].T, sci_masked_1d_noNaN)
    # reconstruct the background based on that vector
    # note that the PCA components WITHOUT masking of the PSF location is being
    # used to reconstruct the background
    recon_2d = np.dot(pca_cube[0:n_PCA,:,:].T, soln_vector[0]).T

    if subt_median:
        # add the offset frame back in
        recon_2d = np.add(recon_2d,raw_pca_training_median)

    # also return the PCA components WITH masking of the PSF location
    recon_2d_masked = np.multiply(recon_2d,mask_weird)

    d = {'pca_vector': soln_vector[0], 'recon_2d': recon_2d, 'recon_2d_masked': recon_2d_masked}

    return d
