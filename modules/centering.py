import multiprocessing
import configparser
import glob
import time
from astropy.io import fits
from astropy.convolution import convolve, Gaussian1DKernel, interpolate_replace_nans
from astropy.modeling import models, fitting
from modules import *

import matplotlib
matplotlib.use('agg') # avoids some crashes when multiprocessing
import matplotlib.pyplot as plt


class Centering:
    '''
    Centering of cookie cut-out frames
    '''

    def __init__(self, config_data=config):

        self.config_data = config_data


    def __call__(self, abs_sci_name):
        '''
        Actual centering, for a single frame so as to parallelize job

        INPUTS:
        sci_name: science array filename
        '''

        # read in the cutout science frame
        sci, header_sci = fits.getdata(abs_sci_name, 0, header=True)


        # get coordinate grid info
        y, x = np.mgrid[0:np.shape(sci)[0],0:np.shape(sci)[1]]
        z = np.copy(sci)

        # make an initial Gaussian guess
        p_init = models.Gaussian2D(amplitude=60000.,
                                   x_mean=np.float(0.5*np.shape(sci)[1]),
                                   y_mean=np.float(0.5*np.shape(sci)[0]),
                                   x_stddev=6.,
                                   y_stddev=6.)
        fit_p = fitting.LevMarLSQFitter()

        # fit the data
        try:
            p = fit_p(p_init, x, y, z)
            ampl, x_mean, y_mean, x_stdev, y_stdev, theat = p._parameters

        except:
            return

        # get the residual frame
        resids = z - p(x, y)

        # make FYI plots of the data with the best-fit model
        plt.clf()
        plt.figure(figsize=(8, 2.5))
        plt.subplot(1, 3, 1)
        plt.imshow(z, origin='lower', interpolation='nearest', vmin=-10, vmax=60000)
        plt.title("Data")
        plt.subplot(1, 3, 2)
        plt.imshow(p(x, y), origin='lower', interpolation='nearest', vmin=-10, vmax=60000)
        plt.title("Model")
        plt.subplot(1, 3, 3)
        plt.imshow(resids, origin='lower', interpolation='nearest', vmin=-10, vmax=60000)
        plt.title("Residual")
        plt.suptitle("Frame " + os.path.basename(abs_sci_name))
        abs_best_fit_gauss_png = str(self.config_data["data_dirs"]["DIR_FYI_INFO"] + \
                                     '/centering_best_fit_gaussian_' + \
                                     os.path.basename(abs_sci_name).split(".")[0] + '.png')
        plt.savefig(abs_best_fit_gauss_png)
        plt.close()
        plt.clf()      

        # center the frame
        # N.b. for a 100x100 image, the physical center is at Python coordinate (49.5,49.5)
        # i.e., in between pixels 49 and 50 in both dimensions (Python convention),
        # or at coordinate (50.5,50.5) in DS9 convention
        y_true_center = 0.5*np.shape(sci)[0]-0.5
        x_true_center = 0.5*np.shape(sci)[1]-0.5

        # locate bad pixels, so we can shift them and turn their counterparts in the science image into NaNs
        mask_pixels = np.zeros(np.shape(sci))
        mask_pixels[sci == 0] = 1 # these are the bad pixels

        # shift in [+y,+x] convention
        sci_shifted = scipy.ndimage.interpolation.shift(sci,
                                                        shift = [y_true_center-y_mean, x_true_center-x_mean],
                                                        mode = "constant",
                                                        cval = 0.0)

        # shift the mask, too
        mask_pixels_shifted = scipy.ndimage.interpolation.shift(mask_pixels,
                                                                shift = [y_true_center-y_mean, x_true_center-x_mean],
                                                                order = 0,
                                                                mode = "constant",
                                                                cval = 1.0)

        '''
        # turn unphysical pixels to NaNs
        # the below awkwardness is necessary to get the right pixels to be NaNs
        # (unphysical pixels at this stage should be approx -999999)
        cookie_mask1 = np.zeros(np.shape(mask_pixels_shifted))
        cookie_mask1[mask_pixels_shifted == 1] = np.nan
        sci_shifted = np.add(sci_shifted,cookie_mask1)
        # (unphysical pixels at this stage should be NaNs)
        '''
        
        # add a line to the header indicating last reduction step
        header_sci["RED_STEP"] = "cookie_centered"
        print(x_stdev)
        print(header_sci)
        # add more lines to record the parameters and residuals with the Gaussian fit
        header_sci["GAU_XSTD"] = x_stdev # x-width of the Gaussian
        header_sci["GAU_YSTD"] = y_stdev # y-width of the Gaussian
        header_sci["RESD_AVG"] = np.nanmean(np.abs(resids)) # mean of the abs. val. of the 2D residual array
        header_sci["RESD_MED"] = np.nanmedian(np.abs(resids)) # median of the abs. val. of the 2D residual array
        header_sci["RESD_INT"] = np.nansum(np.abs(resids)) # summed (integrated) abs. val. of residuals


        # save a FITS file of the residuals (FYI only)
        fits_residual_frame = str(self.config_data["data_dirs"]["DIR_FYI_INFO"] + \
                                     '/centering_best_fit_gaussian_resids_' + \
                                     os.path.basename(abs_sci_name).split(".")[0] + '.fits')
        fits.writeto(filename=fits_residual_frame,
                     data=resids,
                     header=header_sci,
                     overwrite=True)
        print("Writing out frame of residuals between centered frame and gaussian for " + os.path.basename(abs_sci_name))

        # write out science file
        abs_image_cookie_centered_name = str(self.config_data["data_dirs"]["DIR_CENTERED"] + \
                                             os.path.basename(abs_sci_name))
        fits.writeto(filename=abs_image_cookie_centered_name,
                     data=sci_shifted,
                     header=header_sci,
                     overwrite=True)
        print("Writing out centered frame " + os.path.basename(abs_sci_name))


def main():
    '''
    Center all the PSFs
    '''

    # configuration data
    config = configparser.ConfigParser() # for parsing values in .init file
    config.read("modules/config.ini")

    # multiprocessing instance
    pool = multiprocessing.Pool(ncpu)

    # make a list of the PSF cut-outs (crudely centered only)
    cookies_05_directory = str(config["data_dirs"]["DIR_CUTOUTS"])
    cookies_05_name_array = list(glob.glob(os.path.join(cookies_05_directory, "*.fits")))

    center_cookies = Centering()
    pool.map(center_cookies, cookies_05_name_array)   
