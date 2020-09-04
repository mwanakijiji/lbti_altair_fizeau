# The high-level script for running the Altair/Fizeau reduction pipeline
# Between the below steps, data is transferred in the form of FITS or pickle files,
# and a hash text is written when a module is complete
# In the meantime, FYI plots/tables are written out

import datetime
from modules import *
from modules import (basic_red,
                     fits_hdr,
                     centering,
                     psf_pca_bases,
                     injection_ADI,
                     detection,
                     sensitivity,
                     convert_contrast_limits_to_masses,
                     lambda_over_B_KS_test,
                     lambda_over_B_contrast_curves)

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

## ## FAKE PLANET INJECTION, ADI, DETECTION
injection_ADI.main(inject_iteration=0) # finishes by writing out the median ADI frame
print("altair_pipeline: "+str(datetime.datetime.now())+\
    " Finished injection_ADI.main() iteration 0")

## ## DETECTION
detection.main(inject_iteration=0)
print("altair_pipeline: "+str(datetime.datetime.now())+\
    " Finished detection.main() iteration 0")

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

    # time info
    print("altair_pipeline: "+str(datetime.datetime.now())+\
        " Finished injection_ADI.main() iteration "+str(int(iter_num))+" at "+\
        str(np.round(elapsed_time_iteration))+" walltime since start.")

    # re-check signal, amplitudes
    detection.main(inject_iteration=iter_num)

    # time info
    elapsed_time_iteration = np.subtract(time.time(),start_time)
    print("altair_pipeline: "+str(datetime.datetime.now())+\
        " Finished detection.main() iteration "+str(int(iter_num))+" at "+\
        str(np.round(elapsed_time_iteration))+" walltime since start.")
    print("-"*prog_bar_width)

    # condition for convergence: once crossover changes sign around desired S/N,
    # or we reach iteration number X
    iter_num += 1

# interpolate amplitudes

# median along azimuth

# lambda/D 1-D contrast curve from csv info
sensitivity.main(small_angle_correction=True)

# lambda/D mass limits
convert_contrast_limits_to_masses.main(regime = "lambda_over_D",classical=False)

# lambda/B cross-sections
'''
lambda_over_B_KS_test.main(stripe_w_planet = "0",half_w_planet = "E",write_csv_basename = "junk_test_0E.csv")
'''
lambda_over_B_contrast_curves.main(stripe_w_planet = "0",half_w_planet = "E",read_csv_basename = "test_0E.csv")
lambda_over_B_KS_test.main(stripe_w_planet = "0",half_w_planet = "W",write_csv_basename = "test_0W.csv")
lambda_over_B_contrast_curves.main(stripe_w_planet = "0",half_w_planet = "W",read_csv_basename = "test_0W.csv")

lambda_over_B_KS_test.main(stripe_w_planet = "0V",half_w_planet = "N",write_csv_basename = "junk_test_0VN.csv")
lambda_over_B_contrast_curves.main(stripe_w_planet = "0V",half_w_planet = "N",read_csv_basename = "test_0VN.csv")
lambda_over_B_KS_test.main(stripe_w_planet = "0V",half_w_planet = "S",write_csv_basename = "test_0VS.csv")
lambda_over_B_contrast_curves.main(stripe_w_planet = "0V",half_w_planet = "S",read_csv_basename = "test_0VS.csv")

lambda_over_B_KS_test.main(stripe_w_planet = "1",half_w_planet = "E",write_csv_basename = "test_1E.csv")
lambda_over_B_contrast_curves.main(stripe_w_planet = "1",half_w_planet = "E",read_csv_basename = "test_1E.csv")
lambda_over_B_KS_test.main(stripe_w_planet = "1",half_w_planet = "W",write_csv_basename = "test_1W.csv")
lambda_over_B_contrast_curves.main(stripe_w_planet = "1",half_w_planet = "W",read_csv_basename = "test_1W.csv")

lambda_over_B_KS_test.main(stripe_w_planet = "1V",half_w_planet = "N",write_csv_basename = "test_1VN.csv")
lambda_over_B_contrast_curves.main(stripe_w_planet = "1V",half_w_planet = "N",read_csv_basename = "test_1VN.csv")
lambda_over_B_KS_test.main(stripe_w_planet = "1V",half_w_planet = "S",write_csv_basename = "test_1VS.csv")
lambda_over_B_contrast_curves.main(stripe_w_planet = "1V",half_w_planet = "S",read_csv_basename = "test_1VS.csv")

lambda_over_B_KS_test.main(stripe_w_planet = "2",half_w_planet = "E",write_csv_basename = "test_2E.csv")
lambda_over_B_contrast_curves.main(stripe_w_planet = "2",half_w_planet = "E",read_csv_basename = "test_2E.csv")
lambda_over_B_KS_test.main(stripe_w_planet = "2",half_w_planet = "W",write_csv_basename = "test_2W.csv")
lambda_over_B_contrast_curves.main(stripe_w_planet = "2",half_w_planet = "W",read_csv_basename = "test_2W.csv")

lambda_over_B_KS_test.main(stripe_w_planet = "2V",half_w_planet = "N",write_csv_basename = "test_2VN.csv")
lambda_over_B_contrast_curves.main(stripe_w_planet = "2V",half_w_planet = "N",read_csv_basename = "test_2VN.csv")
lambda_over_B_KS_test.main(stripe_w_planet = "2V",half_w_planet = "S",write_csv_basename = "test_2VS.csv")
lambda_over_B_contrast_curves.main(stripe_w_planet = "2V",half_w_planet = "S",read_csv_basename = "test_2VS.csv")

lambda_over_B_KS_test.main(stripe_w_planet = "3",half_w_planet = "E",write_csv_basename = "test_3E.csv")
lambda_over_B_contrast_curves.main(stripe_w_planet = "3",half_w_planet = "E",read_csv_basename = "test_3E.csv")
lambda_over_B_KS_test.main(stripe_w_planet = "3",half_w_planet = "W",write_csv_basename = "test_3W.csv")
lambda_over_B_contrast_curves.main(stripe_w_planet = "3",half_w_planet = "W",read_csv_basename = "test_3W.csv")

lambda_over_B_KS_test.main(stripe_w_planet = "3V",half_w_planet = "N",write_csv_basename = "test_3VN.csv")
lambda_over_B_contrast_curves.main(stripe_w_planet = "3V",half_w_planet = "N",read_csv_basename = "test_3VN.csv")
lambda_over_B_KS_test.main(stripe_w_planet = "3V",half_w_planet = "S",write_csv_basename = "test_3VS.csv")
lambda_over_B_contrast_curves.main(stripe_w_planet = "3V",half_w_planet = "S",read_csv_basename = "test_3VS.csv")

lambda_over_B_KS_test.main(stripe_w_planet = "4",half_w_planet = "E",write_csv_basename = "test_4E.csv")
lambda_over_B_contrast_curves.main(stripe_w_planet = "4",half_w_planet = "E",read_csv_basename = "test_4E.csv")
lambda_over_B_KS_test.main(stripe_w_planet = "4",half_w_planet = "W",write_csv_basename = "test_4W.csv")
lambda_over_B_contrast_curves.main(stripe_w_planet = "4",half_w_planet = "W",read_csv_basename = "test_4W.csv")

lambda_over_B_KS_test.main(stripe_w_planet = "4V",half_w_planet = "N",write_csv_basename = "test_4VN.csv")
lambda_over_B_contrast_curves.main(stripe_w_planet = "4V",half_w_planet = "N",read_csv_basename = "test_4VN.csv")
lambda_over_B_KS_test.main(stripe_w_planet = "4V",half_w_planet = "S",write_csv_basename = "test_4VS.csv")
lambda_over_B_contrast_curves.main(stripe_w_planet = "4V",half_w_planet = "S",read_csv_basename = "test_4VS.csv")

# lambda/D mass limits
convert_contrast_limits_to_masses.main(regime = "lambda_over_B")
'''
