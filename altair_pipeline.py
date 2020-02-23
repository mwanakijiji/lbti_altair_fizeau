# The high-level script for running the Altair/Fizeau reduction pipeline
# Between the below steps, data is transferred in the form of FITS or pickle files,
# and a hash text is written when a module is complete
# In the meantime, FYI plots/tables are written out

from modules import *
from modules import (basic_red,
                     fits_hdr,
                     centering,
                     psf_pca_bases,
                     injection_ADI,
                     detection,
                     sensitivity)

## ## READ IN HASHABLE CONFIG FILE FOR REDUCTION PARAMETERS: GO AHEAD
## ## WITH BASIC REDUCTIONS, OR SKIP THEM? ETC.

start_time = time.time()

## ## MAKE NEEDED DIRECTORIES
make_dirs()
'''
## ## FITS HEADER METADATA EXTRACTION
fits_meta_data = fits_hdr.main()

## ## BASIC REDUCTIONS
basic_red.main()

## ## CENTERING OF PSFS
centering.main()

## ## PSF PCA BASIS GENERATION
psf_pca_bases.main()
'''
## ## FAKE PLANET INJECTION, ADI, DETECTION
injection_ADI.main(inject_iteration=0) # finishes by writing out the median ADI frame

print("TEST FYI only: between the injection_ADI and detection modules")

## ## DETECTION
detection.main(inject_iteration=0)

print("Total time:")
elapsed_time_iteration = np.subtract(time.time(),start_time)
print(np.round(elapsed_time_iteration))

## ## DETERMINE AMPLITUDES OF COMPANIONS TO GIVE S/N=5
iter_num = 1
while True:
    # Read in detection csv, check S/N (or FPF? maybe I should add that to csv)
    # for each fake companion.
    # Companion-by-companion, change fake companion amplitude by del_X / del_Y /
    # del_Z etc. with sign depending on starting S/N.
    # Then re-inject and re-reduce ADI
    injection_ADI.main(inject_iteration=iter_num)

    # re-check signal, amplitudes
    detection.main(inject_iteration=iter_num)

    # time info
    elapsed_time_iteration = np.subtract(time.time(),start_time)
    print("Done with iter_num:")
    print(iter_num)
    print("Time since start:")
    print(np.round(elapsed_time_iteration))
    print("-"*prog_bar_width)

    # condition for convergence: once crossover changes sign around desired S/N,
    # or we reach iteration number X
    iter_num += 1
'''
# interpolate amplitudes

# median along azimuth

# incorporate
sensitivity.main() # produces 1-D contrast curve from csv info
'''
