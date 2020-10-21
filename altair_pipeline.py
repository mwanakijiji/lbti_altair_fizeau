# The high-level script for running the Altair/Fizeau reduction pipeline
# Between the below steps, data is transferred in the form of FITS or pickle files,
# and a hash text is written when a module is complete
# In the meantime, FYI plots/tables are written out

import datetime
from modules import *
from modules import (convert_contrast_limits_to_masses,
                    lambda_over_B_KS_test,
                     lambda_over_B_contrast_curves)

## ## READ IN HASHABLE CONFIG FILE FOR REDUCTION PARAMETERS: GO AHEAD
## ## WITH BASIC REDUCTIONS, OR SKIP THEM? ETC.

start_time = time.time()
'''
## ## MAKE NEEDED DIRECTORIES
make_dirs()

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

# lambda/B tests
'''
#lambda_over_B_KS_test.main(stripe_w_planet = "0",half_w_planet = "E",write_csv_basename = "test01_20201021_0E.csv")
lambda_over_B_KS_test.main(stripe_w_planet = "0",half_w_planet = "W",write_csv_basename = "test01_20201021_0W.csv")
#lambda_over_B_KS_test.main(stripe_w_planet = "0V",half_w_planet = "N",write_csv_basename = "test01_20201021_0VN.csv")
#lambda_over_B_KS_test.main(stripe_w_planet = "0V",half_w_planet = "S",write_csv_basename = "test01_20201021_0VS.csv")

#lambda_over_B_KS_test.main(stripe_w_planet = "1",half_w_planet = "E",write_csv_basename = "test01_20201021_1E.csv")
#lambda_over_B_KS_test.main(stripe_w_planet = "1",half_w_planet = "W",write_csv_basename = "test01_20201021_1W.csv")
#lambda_over_B_KS_test.main(stripe_w_planet = "1V",half_w_planet = "N",write_csv_basename = "test01_20201021_1VN.csv")
#lambda_over_B_KS_test.main(stripe_w_planet = "1V",half_w_planet = "S",write_csv_basename = "test01_20201021_1VS.csv")

#lambda_over_B_KS_test.main(stripe_w_planet = "2",half_w_planet = "E",write_csv_basename = "test01_20201021_2E.csv")
#lambda_over_B_KS_test.main(stripe_w_planet = "2",half_w_planet = "W",write_csv_basename = "test01_20201021_2W.csv")
#lambda_over_B_KS_test.main(stripe_w_planet = "2V",half_w_planet = "N",write_csv_basename = "test01_20201021_2VN.csv")
#lambda_over_B_KS_test.main(stripe_w_planet = "2V",half_w_planet = "S",write_csv_basename = "test01_20201021_2VS.csv")

#lambda_over_B_KS_test.main(stripe_w_planet = "3",half_w_planet = "E",write_csv_basename = "test01_20201021_3E.csv")
#lambda_over_B_KS_test.main(stripe_w_planet = "3",half_w_planet = "W",write_csv_basename = "test01_20201021_3W.csv")
#lambda_over_B_KS_test.main(stripe_w_planet = "3V",half_w_planet = "N",write_csv_basename = "test01_20201021_3VN.csv")
#lambda_over_B_KS_test.main(stripe_w_planet = "3V",half_w_planet = "S",write_csv_basename = "test01_20201021_3VS.csv")

#lambda_over_B_KS_test.main(stripe_w_planet = "4",half_w_planet = "E",write_csv_basename = "test01_20201021_4E.csv")
#lambda_over_B_KS_test.main(stripe_w_planet = "4",half_w_planet = "W",write_csv_basename = "test01_20201021_4W.csv")
#lambda_over_B_KS_test.main(stripe_w_planet = "4V",half_w_planet = "N",write_csv_basename = "test01_20201021_4VN.csv")
#lambda_over_B_KS_test.main(stripe_w_planet = "4V",half_w_planet = "S",write_csv_basename = "test01_20201021_4VS.csv")


#local_stem = "/Users/nyumbani/Documents/git.repos/lbti_altair_fizeau/" # string for writing/reading in KS data
local_stem = "./" # string for writing/reading in KS data
#lambda_over_B_contrast_curves.main(read_csvs_directory = local_stem+"20201021_allE_run01/", write_stem = "20201021_allE")
#lambda_over_B_contrast_curves.main(read_csvs_directory = local_stem+"20201021_allW_run01/", write_stem = "20201021_allW")
#lambda_over_B_contrast_curves.main(read_csvs_directory = local_stem+"20201021_allN_run01/", write_stem = "20201021_allN")
#lambda_over_B_contrast_curves.main(read_csvs_directory = local_stem+"20201021_allS_run01/", write_stem = "20201021_allS")

'''
# lambda/D mass limits
convert_contrast_limits_to_masses.main(regime = "lambda_over_B")
'''
