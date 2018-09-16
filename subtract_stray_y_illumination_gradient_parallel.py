# There is a stray "illumination" gradient in y in the dark-subtracted LMIRcam readouts
# from the Altair data. It's not physical illumination, but apparently has something to
# do with resetting of the detector (see emails from Jarron Leisenring and Jordan Stone,
# Sept. 5/6 2018). It's consistent, so I'll try to subtract it out

# Created 2018 Sept. 6 by E.S.

import numpy as np
import matplotlib.pyplot as plt
import scipy
from astropy.io import fits
from astropy.convolution import convolve, Gaussian1DKernel
import multiprocessing as mp
from multiprocessing import Process, Queue, Pool

# stem 
stem = ('/home/../../media/unasemaje/Elements/lbti_data_reduction/180507_fizeau_altair/01_fix_pixed/second_attempt/')

gauss_kernel = Gaussian1DKernel(5)

def remove_ramp(framenum):
    
    img_string = 'lm_180507_'+str("{:0>6d}".format(framenum))+'.fits'
    
    # if file exists
    if os.path.isfile(stem+img_string):
        
        # read in image
        sciImg, header = fits.getdata(stem+img_string,0,header=True)

        # find the median in x across the whole array as a function of y
        stray_ramp = np.nanmedian(sciImg[:,:],axis=1)

        # smooth it
        smoothed_stray_ramp = convolve(stray_ramp, gauss_kernel)

        # subtract from the whole array (note there will still be residual channel pedestals)
        sciImg1 = np.subtract(sciImg,np.tile(smoothed_stray_ramp,(2048,1)).T)

        # write back out
        hdu = fits.PrimaryHDU(sciImg1.astype(np.float32), header=header)
        hdul = fits.HDUList([hdu])
    
        hdul.writeto(stem+'../../02_stray_ramp_removed/'+img_string, overwrite=True)
        print('Saved '+str("{:0>6d}".format(framenum)))
        
    else:
        
        print('File '+img_string+' not found')


# parallelize it
framenumArray = np.arange(4249,11336)
ncpu = mp.cpu_count()
pool = Pool(ncpu) # create pool object
mapping = pool.map(remove_ramp,framenumArray)

##################

# do it!
if __name__ == '__main__':
    main()

