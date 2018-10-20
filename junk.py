
# coding: utf-8

# In[1]:

# This makes a first contrast curve for Altair

# created 2018 Sept. 20 by E.S.


# In[1]:

import urllib
import numpy as np
import matplotlib.pyplot as plt

import PynPoint

from PynPoint import Pypeline
from PynPoint.IOmodules.Hdf5Reading import Hdf5ReadingModule
from PynPoint.IOmodules.FitsWriting import FitsWritingModule
from PynPoint.IOmodules.FitsReading import FitsReadingModule
from PynPoint.ProcessingModules import PSFpreparationModule,                                        PcaPsfSubtractionModule,                                        ContrastCurveModule,                                        FluxAndPosition


# In[17]:

working_place = "./pynpoint_experimentation_altair/working_place/"
input_place = "./pynpoint_experimentation_altair/input_place/"
output_place = "./pynpoint_experimentation_altair/output_place/"

#url = urllib.URLopener()
#url.retrieve("https://people.phys.ethz.ch/~stolkert/BetaPic_NACO_Mp.hdf5",
#             input_place+"BetaPic_NACO_Mp.hdf5")

pipeline = Pypeline(working_place_in=working_place,
                    input_place_in=input_place,
                    output_place_in=output_place)

# now a *.ini file has been generated (this includes the PIXSCALE), if no pre-existing one was there


# In[ ]:

## NOW, EDIT THE *INI FILE FOR 

## PIXSCALE = 0.0107


# In[18]:

# read in science FITS files

read_science = FitsReadingModule(name_in="read_science",
                                 input_dir=None,
                                 image_tag="science",
                                 check=True)

pipeline.add_module(read_science)


# In[19]:

# read in PSF reference FITS files (i.e., unsaturated frames)

read_ref_psf = FitsReadingModule(name_in="read_ref_psf",
                                 input_dir=input_place+'ref_psf/',
                                 image_tag="ref_psf",
                                 check=True)

pipeline.add_module(read_ref_psf)


# In[ ]:

## might insert PSF Preparation here later, to upsample and normalize (but not mask! let contrast curve module do that)


# In[ ]:

# make a contrast curve

'''
cent_size: mask radius
'''

contrast_curve = ContrastCurveModule(name_in="contrast_curve",
                            image_in_tag="prep",
                            psf_in_tag="model_psf",
                            contrast_out_tag="contrast_landscape",
                            pca_out_tag="pca_resids",
                            pca_number=20,
                            psf_scaling=1,
                            separation=(0.1, 1.0, 0.1), 
                            angle=(0.0, 360.0, 60.0), 
                            magnitude=(7.5, 1.0),
                            cent_size=None)

pipeline.add_module(contrast_curve)


# In[ ]:

contrast_curve_results = pipeline.get_data("contrast_landscape")


# In[ ]:

# contrast curve

# [0]: separation
# [1]: azimuthally averaged contrast limits
# [2]: the azimuthal variance of the contrast limits
# [3]: threshold of the false positive fraction associated with sigma

contrast_curve_results


# In[ ]:

# make plots

f, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(contrast_curve_results[:,0],contrast_curve_results[:,1])
ax1.set_title('Azim. averaged contrast limit')
ax1.set_xlabel('Radius (asec)')
ax2.set_ylabel('del_mag')
ax2.plot(contrast_curve_results[:,0],contrast_curve_results[:,3])
ax2.set_title('Threshold of FPF')
ax2.set_xlabel('Radius (asec)')
f.savefig('test.png')


# In[ ]:

##########################


# In[5]:

# inject fake planets

'''
inject = FluxAndPosition.FakePlanetModule(position=(0.46,0),
                          magnitude=0,
                          psf_scaling=1,
                          name_in="inject",
                          image_in_tag="stack",
                          psf_in_tag="model_psf",
                          image_out_tag="fake_planet_output")

pipeline.add_module(inject)
'''


# In[6]:

# write out science frames with fake planet

'''
write_inject = FitsWritingModule(file_name="junk_stack_fake_planet.fits",
                              name_in="write_inject",
                              output_dir=output_place,
                              data_tag="fake_planet_output")

pipeline.add_module(write_inject)
'''


# In[20]:

# prepare psf


prep_fake_planet = PSFpreparationModule(name_in="prep_fake_planet",
                            image_in_tag="stack",
                            image_out_tag="prep",
                            image_mask_out_tag=None,
                            mask_out_tag=None,
                            norm=False,
                            resize=None,
                            cent_size=0.15,
                            edge_size=1.1)

pipeline.add_module(prep_fake_planet)


# In[21]:

pipeline.run()

# after running the pipeline, hdf5 attributes like PIXSCALE for groups
# other than /config/ should be available


# In[22]:

pixscale_config = pipeline.get_attribute("config", "PIXSCALE")
print(pixscale_config)
pixscale_prep = pipeline.get_attribute("prep", "PIXSCALE")
print(pixscale_prep)


# In[7]:

print(pixscale_config)


# In[8]:

# write out prepped psf

'''
# checks HDF5 file
write_prep = FitsWritingModule(file_name="junk_prep.fits",
                              name_in="write_prep",
                              output_dir=output_place,
                              data_tag="prep")

#pipeline.add_module(write_prep)
'''


# In[7]:

# do PCA PSF subtraction

'''
pca = PcaPsfSubtractionModule(pca_numbers=(5, ),
                              name_in="pca",
                              images_in_tag="fake_planet_output",
                              reference_in_tag="fake_planet_output",
                              res_mean_tag="mean_residuals",
                              res_median_tag="median_residuals",
                              res_arr_out_tag="all_resids",
                              res_rot_mean_clip_tag="resid_rot",
                              verbose=True)

pipeline.add_module(pca)

# note:
# images_in_tag: science images
# reference_in_tag: reference images, which COULD be the science images
'''


# In[7]:

## THIS IS TEST ONLY ## do PCA PSF subtraction

'''
pca = PcaPsfSubtractionModule(pca_numbers=(5, ),
                              name_in="pca",
                              images_in_tag="read_science",
                              reference_in_tag="read_science",
                              res_mean_tag="mean_residuals",
                              res_median_tag="median_residuals",
                              res_arr_out_tag="all_resids",
                              res_rot_mean_clip_tag="resid_rot",
                              verbose=True)

pipeline.add_module(pca)
'''
# note:
# images_in_tag: science images
# reference_in_tag: reference images, which COULD be the science images


# In[8]:

# write out outputs from PCA PSF subtraction

'''
# checks HDF5 file
read_test1 = FitsWritingModule(file_name="junk_mean_residuals.fits",
                              name_in="read_test1",
                              output_dir=output_place,
                              data_tag="mean_residuals")

pipeline.add_module(read_test1)
'''


# In[9]:

# write out outputs from PCA PSF subtraction

'''
# checks HDF5 file
read_test2 = FitsWritingModule(file_name="junk_median_residuals.fits",
                              name_in="read_test2",
                              output_dir=output_place,
                              data_tag="median_residuals")

pipeline.add_module(read_test2)
'''


# In[10]:

# write out outputs from PCA PSF subtraction

'''
# checks HDF5 file
read_test4 = FitsWritingModule(file_name="junk_resid_rot.fits",
                              name_in="read_test4",
                              output_dir=output_place,
                              data_tag="resid_rot")

pipeline.add_module(read_test4)
'''


# In[6]:

# make a contrast curve

'''
cent_size: mask radius
'''

contrast_curve = ContrastCurveModule(name_in="contrast_curve",
                            image_in_tag="prep",
                            psf_in_tag="model_psf",
                            contrast_out_tag="contrast_landscape",
                            pca_out_tag="pca_resids",
                            pca_number=20,
                            psf_scaling=1,
                            separation=(0.1, 1.0, 0.1), 
                            angle=(0.0, 360.0, 60.0), 
                            magnitude=(7.5, 1.0),
                            cent_size=None)

pipeline.add_module(contrast_curve)


# In[9]:

pipeline.run()


# In[11]:

contrast_curve_results = pipeline.get_data("contrast_landscape")
#residuals = pipeline.get_data("residuals")
#pixscale = pipeline.get_attribute("stack", "PIXSCALE")

#size = pixscale*np.size(residuals, 1)/2.


# In[12]:

# contrast curve

# [0]: separation
# [1]: azimuthally averaged contrast limits
# [2]: the azimuthal variance of the contrast limits
# [3]: threshold of the false positive fraction associated with sigma

contrast_curve_results


# In[20]:



f, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(contrast_curve_results[:,0],contrast_curve_results[:,1])
ax1.set_title('Azim. averaged contrast limit')
ax1.set_xlabel('Radius (asec)')
ax2.set_ylabel('del_mag')
ax2.plot(contrast_curve_results[:,0],contrast_curve_results[:,3])
ax2.set_title('Threshold of FPF')
ax2.set_xlabel('Radius (asec)')
f.savefig('test.png')


# In[19]:

plt.imshow(residuals[0, ], origin='lower', extent=[size, -size, -size, size])
plt.title("beta Pic b - NACO M' - mean residuals")
plt.xlabel('R.A. offset [arcsec]', fontsize=12)
plt.ylabel('Dec. offset [arcsec]', fontsize=12)
plt.colorbar()
#plt.show()
plt.savefig(output_place+"residuals.png")


# In[13]:

pipeline


# In[19]:

np.size(residuals)


# In[10]:

pipeline.add_module(writefits)


# In[11]:

writefits.run()


# In[ ]:




