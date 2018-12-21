#!/usr/bin/python
# This is a simple script to set camera parameters, take data, and offset the telescope
# This is a *dual aperture* script

# Operator must set exposure time and number of coadds first!
# must begin with the stars in the *UP* nod position

import sys, os, string, time, pyfits,  pdb, copy
import numpy as np
from pyindi import * 
from scipy import ndimage, sqrt, stats
import matplotlib.pyplot as plt

# do simple for-loops to move the FPC and HPC

pi = PyINDI()

#pi.setINDI("LMIRCAM.Command.text=1 savedata")

#pi.setINDI("LMIRCAM.Command.text=go")
'''
for step in range(0,4):
    pi.setINDI("Acromag.HPC.Tip=0;Tilt=0.5;Piston=0;Mode=1")
    time.sleep(1.0)
    pi.setINDI("LMIRCAM.Command.text=go")
    pi.setINDI("Acromag.HPC.Tip=-0.5;Tilt=0.0;Piston=0;Mode=1")
    time.sleep(1.0)
    pi.setINDI("LMIRCAM.Command.text=go")
    pi.setINDI("Acromag.HPC.Tip=0;Tilt=-0.5;Piston=0;Mode=1")
    time.sleep(1.0)
    pi.setINDI("LMIRCAM.Command.text=go")
    pi.setINDI("Acromag.HPC.Tip=0.5;Tilt=0.0;Piston=0;Mode=1")
    time.sleep(1.0)
    pi.setINDI("LMIRCAM.Command.text=go")
    pi.setINDI("Acromag.FPC.Tip=0.0;Tilt=0.5;Piston=0;Mode=1")
    time.sleep(1.0)
    pi.setINDI("LMIRCAM.Command.text=go")
    pi.setINDI("Acromag.FPC.Tip=-0.5;Tilt=0;Piston=0;Mode=1")
    time.sleep(1.0)
    pi.setINDI("LMIRCAM.Command.text=go")
    pi.setINDI("Acromag.FPC.Tip=0.0;Tilt=-0.5;Piston=0;Mode=1")
    time.sleep(1.0)
    pi.setINDI("LMIRCAM.Command.text=go")
    pi.setINDI("Acromag.FPC.Tip=0.5;Tilt=0;Piston=0;Mode=1")
    time.sleep(1.0)
    pi.setINDI("LMIRCAM.Command.text=go")
'''

#####################################################
# START NEW CODE

# subtract mean or median (method) of the image
# w = edge px on either side
def bkgdsub(image, method):

    image[image==0] = numpy.nan

    ctr = int( np.floor( image.shape[0] / 2. ) ) # half-width

    tmpimg = copy.copy(image) # make intermediary image

    # take mean or median across rows (for removing horizontal striping of array)
    if method == 'mean':
        #rowbkgd = stats.nanmean(tmpimg,1) # if deprecated, use 
        rowbkgd = np.nanmean(tmpimg,1)
    elif method == 'median':
        rowbkgd = np.nanmedian(tmpimg,1)
    rowbkgd2d = np.tile(np.reshape(rowbkgd,[len(rowbkgd),1]),[1,image.shape[0]]) # paint this column of median values into a 2048x2048 array
    tmpimg = tmpimg - rowbkgd2d # simple form of background subtraction (vertical striping between columns will still be there)

    # do same as above, but for the columns
    if method == 'mean':
        colbkgd = np.nanmean(tmpimg,0)
    elif method == 'median':
        colbkgd = np.nanmedian(tmpimg,0)
    colbkgd2d = numpy.tile(np.reshape(colbkgd,[1,len(colbkgd)]),[image.shape[1],1])
    image = image - rowbkgd2d - colbkgd2d

    image[np.isnan(image)] = 0

    # image now should have a nice flat background, but bad pixels will remain
    return image


# process the image
def processImg(imgDummy, methodDummy):
    
    # bias level correction
    imgSub = bkgdsub(imgDummy,methodDummy) # simple background smoothing
    imgSub -= np.median(imgSub) # subtract residual background
    imgSubM = ndimage.median_filter(imgSub,3) # smoothed image

    # define BP mask
    imgDiff = numpy.abs(imgSub - imgSubM) # difference between smoothed and unsmoothed images
    stddev = numpy.std(imgDiff)
    mask = ( imgDiff > 4*stddev ) & ( imgDiff > 0.15 * numpy.abs(imgSub) ) # mask: True=bad, False=good

    imgSubBP = copy.copy(imgSub)
    imgSubBP[mask] = imgSubM[mask] # set bad pixels in unsmoothed image equal to those in smoothed image

    return imgSubBP


# find star and return coordinates [y,x]
def findStar(image):
    if autoFindStar:

        #imageThis = numpy.copy(image)

        '''
        if (PSFside == 'left'):
            imageThis[:,1024:-1] = 0
        elif (PSFside == 'right'):
            imageThis[:,0:1024] = 0
        '''

        imageG = ndimage.gaussian_filter(image, 6) # further remove effect of bad pixels (somewhat redundant?)
        loc = numpy.argwhere(imageG==imageG.max())
        cx = loc[0,1]
        cy = loc[0,0]

        #plt.imshow(imageG, origin="lower")
        #pdb.set_trace()
        #plt.scatter([cx,cx],[cy,cy], c='r', s=50)
        #plt.colorbar()
        #plt.show()
        #print [cy, cx] # check

    return [cy, cx]


def dist_pix(current,goal):

    dist = np.sqrt(np.power((current[1]-goal[1]),2) + np.power((current[0]-goal[0]),2) )

    return dist


###############################
## END FUNCTION DEFINITIONS

autoFindStar = True  # auto-detect star in frame?

## SET PIXEL LOCATION I WANT PSFS TO BE 
psf_loc_setpoint = [1220,800]

### MOVE IN HALF-MOON TO SEE SX FIRST
pi.setINDI("Lmir.lmir_FW2.command", 'SX-Half-moon', wait=True)#, timeout=45, wait=True)

### CHANGE THIS TO WHILE-LOOP LATER
while True: # do three iterations to try to get SX PSF on the same pixel

    # locate SX PSF
    f=pi.getFITS("LMIRCAM.DisplayImage.File", "LMIRCAM.GetDisplayImage.Now", wait=True) # get what LMIR is seeing
    imgb4 = f[0].data
    #imgb4bk = bkgdsub(imgb4,'median') # simple background smoothing
    #imgb4bk -= numpy.median(imgb4bk) # subtract residual background

    imgb4bk = processImg(imgb4, 'median') # return background-subtracted, bad-pix-corrected image
    psf_loc = findStar(imgb4bk) # locate the PSF
    
    ### MOVE FPC IN ONE STEP TO MOVE PSF TO RIGHT LOCATION
    vector_move_pix = np.subtract(psf_loc_setpoint,psf_loc) # vector of required movement in pixel space
    vector_move_asec = np.multiply(vector_move_pix,0.0107) # convert to asec
    pi.setINDI("Acromag.FPC.Tip="+'{0:.1f}'.format(vector_move_asec[0])+";Tilt="+'{0:.1f}'.format(vector_move_asec[1])+";Piston=0;Mode=1")
    
    ### RE-LOCATE SX PSF; CORRECTION NEEDED?
    f=pi.getFITS("LMIRCAM.DisplayImage.File", "LMIRCAM.GetDisplayImage.Now", wait=True) # get what LMIR is seeing
    imgb4 = f[0].data
    imgb4bk = processImg(imgb4, 'median') # return background-subtracted, bad-pix-corrected image
    psf_loc = findStar(imgb4bk) # locate the PSF
 
    print('-------------------')
    print('PSF location setpoint:')
    print(psf_loc_setpoint) 
    print('Current PSF loc:') 
    print(psf_loc)
    
    if (dist_pix(psf_loc,psf_loc_setpoint) < 5.):
        print('-------------------')
        print('Done moving one side. Switching to the other side.')
        break 

    print('Moving PSF again...')
    
    
### MOVE IN HALF-MOON TO SEE DX NEXT
pi.setINDI("Lmir.lmir_FW2.command", 'DX-Half-moon', wait=True)
   
    
### NEW FOR-LOOP HERE
while True: # do three iterations to try to get SX PSF on the same pixel
    
    # locate DX PSF
    f=pi.getFITS("LMIRCAM.DisplayImage.File", "LMIRCAM.GetDisplayImage.Now", wait=True) # get what LMIR is seeing
    imgb4 = f[0].data
                        
    imgb4bk = processImg(imgb4, 'median') # return background-subtracted, bad-pix-corrected image
    psf_loc = findStar(imgb4bk) # locate the PSF
                                
    ### MOVE HPC IN ONE STEP TO MOVE PSF TO RIGHT LOCATION
    vector_move_pix = np.subtract(psf_loc_setpoint,psf_loc) # vector of required movement in pixel space
    vector_move_asec = np.multiply(vector_move_pix,0.0107) # convert to asec
    pi.setINDI("Acromag.HPC.Tip="+'{0:.1f}'.format(vector_move_asec[0])+";Tilt="+'{0:.1f}'.format(vector_move_asec[1])+";Piston=0;Mode=1")
                                                
    ### RE-LOCATE SX PSF; CORRECTION NEEDED?
    f=pi.getFITS("LMIRCAM.DisplayImage.File", "LMIRCAM.GetDisplayImage.Now", wait=True) # get what LMIR is seeing
    imgb4 = f[0].data
    imgb4bk = processImg(imgb4, 'median') # return background-subtracted, bad-pix-corrected image
    psf_loc = findStar(imgb4bk) # locate the PSF
    
    print('-------------------')
    print('PSF location setpoint:')
    print(psf_loc_setpoint) 
    print('Current DX PSF loc:')
    print(psf_loc)
                                                                            
    if (dist_pix(psf_loc,psf_loc_setpoint) < 5.):
        print('-------------------')
        print('Done moving DX. Switching to the other side.')
        break
                                                                                                    
    print('Moving DX again...')    


print('Done moving PSFs. Reopening LMIR FW2.')
pi.setINDI("Lmir.lmir_FW2.command", 'Open', wait=True) 
