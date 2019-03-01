'''
Prepares the data: bad pixel masking, background-subtraction, etc.
## ## This is descended from
## ## find_background_limit.ipynb
## ## subtract stray y illumination gradient parallel.py
## ## make_pca_basis_cube_altair.py
## ## pca_background_subtraction.ipynb
'''


class BasicRed():
    '''
    Carry out basic cleaning of the data
    '''

    def __init__(self):

        pass

    def __call__(self):

        pass

    def dark_subt_gen(self):
        '''
        Generalized dark subtraction
        '''

        pass

    def fix_pix_gen(self):
        '''
        Interpolates over bad pixels ## ## VANESSAS ALGORITHM?
        '''

        pass

    def remove_stray_ramp(self):
        '''
        Removes an additive electronic artifact illumination ramp in y at the top of the 
        LMIRcam readouts. (The ramp has something to do with resetting of the detector while 
        using the 2018A-era electronics; i.e., before MACIE overhaul in summer 2018-- see 
        emails from J. Leisenring and J. Stone, Sept. 5/6 2018)
        '''

        pass

    def pca_background_decomp(self):
        '''
        Generates a PCA cube based on the backgrounds in the science frames.
        '''

        pass

    def pca_background_subt(self):
        '''
        Does a PCA decomposition of a given frame, and subtracts the background
        ## ## N.b. remaining pedestal should be photons alone; how smooth is it?
        '''

        pass

    def calc_noise(self):
        '''
        Finds noise characteristics: where is the background limit? etc.
        '''

        pass

    def cookie_cutout(self):
        '''
        Cuts out region around PSF commensurate with the AO control radius
        '''

        pass
