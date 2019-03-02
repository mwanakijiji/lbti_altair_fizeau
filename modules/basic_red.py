'''
Prepares the data: bad pixel masking, background-subtraction, etc.
## ## This is descended from
## ## find_background_limit.ipynb
## ## subtract stray y illumination gradient parallel.py
## ## make_pca_basis_cube_altair.py
## ## pca_background_subtraction.ipynb
'''

from modules import *
from astropy.io import fits
import multiprocessing
import configparser
import glob


def dark_subt_parallel():
    '''
    Generalized dark subtraction

    INPUTS:
    none; list of science frame filenames is generated internally

    OUTPUTS:
    none; dark-subtracted files are written out
    '''

    # make a list of the science array files
    sci_directory = str(config["data_dir_stem"]["DATA_DIR_STEM"]) + \
              str(config["data_dir_leaves"]["DIR_RAW_DATA_LEAF"])
    sci_name_array = list(glob.glob(os.path.join(sci_directory,"*.fits")))

    #dark_subt = BasicRed().dark_subt_single()

    def dark_subt_single(abs_sci_name):
        '''
        Actual subtraction, for a single frame so as to parallelize job

        INPUTS:
        sci_name: science array filename 
        '''
            
        # read in the science frame from raw data directory
        print(abs_sci_name)

        sci, header_sci = fits.getdata(abs_sci_name,0,header=True)
        print(np.shape(sci))

        # subtract from image; data type should allow negative numbers
        image_dark_subtd = np.subtract(sci,dark).astype(np.int32)

        # add a line to the header indicating last reduction step
        header_sci["LAST_REDUCT_STEP"] = "dark-subtraction"

        # write file out
        abs_image_dark_subtd_name = str(self.config["data_dir_stem"]["DATA_DIR_STEM"]) + \
              str(self.config["data_dir_leaves"]["DIR_DARK_SUBTED_LEAF"] + sci_name)
        fits.writeto(filename = abs_image_dark_subtd_name,
                         data = image_dark_subt,
                         header = header_sci,
                         overwrite=False)
        print("Write out dark-subtracted frame " + sci_name)

    # subtract darks in parallel
    print("Subtracting darks with " + str(ncpu) + " CPUs...")
    pool = multiprocessing.Pool(ncpu)
    outdat = pool.map(dark_subt_single, sci_name_array)
        
    return
