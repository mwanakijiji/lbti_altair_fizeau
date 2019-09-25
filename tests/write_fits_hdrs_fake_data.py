# This reads in fake FITS files and writes them out with headers so as
# to mimic real data.

import os
import glob
import numpy as np
import pandas as pd
from astropy.io import fits

# make list of files
direct_name = "/vol_c/synthetic_fizeau_data/pipeline_06_centered/"
file_name_array = list(glob.glob(os.path.join(direct_name, "*.fits")))

# loop through them
for u in file_name_array:

    test_img, header = fits.getdata(u, 0, header=True)

    '''
    header["RESD_AVG"] = 0
    header["RESD_MED"] = 0
    header["RESD_INT"] = 0
    header["GAU_XSTD"] = 0
    header["GAU_YSTD"] = 0
    header["PCCLOSED"] = 1
    '''
    header["LBT_PARA"] = 40.*np.random.random()
    print(header["LBT_PARA"])

    fits.writeto(filename = "/vol_c/synthetic_fizeau_data/" + os.path.basename(u),
                     data = test_img.astype(np.float32),
                     header = header,
                     overwrite = True)

    print(os.path.basename(u))

