# This is for taking a chunck of slices from a giant cube and saving them for downloading
# and checking 

# created 2019 Sept. 27 by E.S.

import numpy as np
from astropy.io import fits

# read in image
sciImg, header = fits.getdata("junk_cube_pre_removal.fits",0,header=True)
    
sciImg_chunk = sciImg[0:200,:,:]

# write
fits.writeto(filename = "junk_cube_pre_removal_chunk.fits",
             data = sciImg_chunk,
             overwrite = True)
