import pandas as pd
from modules import *

class FitsHdr:
    '''
    Reads in a .csv file of meta-data from the FITS file headers, as written out
    manually with a Jupyter Notebook
    '''

    def __init__(self, config_data=config):

        self.config_data = config_data

        # initialize absolute meta data file name
        self.abs_metadata_csv_name = str(self.config_data["data_dirs"]["DIR_SRC"] + \
          self.config_data["dataset_string"]["DATASET_STRING"] + \
          "_metadata.csv")

    def read_meta_data(self):
        '''
        Read in the .csv
        '''

        # read in the science frame from raw data directory
        fits_hdr_data = pd.read_csv(self.abs_metadata_csv_name)

        return fits_hdr_data


def main():
    '''
    Read in FITS meta-data in the form of a giant table
    '''#

    # configuration data
    config = configparser.ConfigParser() # for parsing values in .init file
    config.read("modules/config.ini")#

    print('returning')
    data = FitsHdr()
    
    return data.read_meta_data()
