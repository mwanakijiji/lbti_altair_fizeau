# The high-level script for running the Altair/Fizeau reduction pipeline
# Between the below steps, data is transferred in the form of FITS or pickle files,
# and a hash text is written when a module is complete
# In the meantime, FYI plots/tables are written out

from modules import *
from modules import basic_red, fits_hdr

## ## READ IN HASHABLE CONFIG FILE FOR REDUCTION PARAMETERS: GO AHEAD
## ## WITH BASIC REDUCTIONS, OR SKIP THEM? ETC.

## ## FITS HEADER METADATA EXTRACTION
fits_meta_data = fits_hdr.main()

## ## BASIC REDUCTIONS
basic_red.main()

## ## CENTERING OF PSFS

## ## PCA_BASIS GENERATION

## ## FAKE PLANET INJECTION

## ## HOST REMOVAL

## ## ADI: large radii, small radii

## ## DETECTION

## ## ORBITAL PARAMETER FORWARD MODELING

## ## SENSITIVITY
