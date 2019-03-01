# The high-level script for running the Altair/Fizeau reduction pipeline
# Between the below steps, data is transferred in the form of FITS or pickle files,
# and a hash text is written when a module is complete
# In the meantime, FYI plots/tables are written out

from modules import *

## ## READ IN HASHABLE CONFIG FILE FOR REDUCTION PARAMETERS: GO AHEAD
## ## WITH BASIC REDUCTIONS, OR SKIP THEM? ETC.

## ## BASIC REDUCTIONS

## ## FITS HEADER METADATA EXTRACTION

## ## CENTERING OF PSFS

## ## PCA_BASIS GENERATION

## ## FAKE PLANET INJECTION

## ## HOST REMOVAL

## ## ADI: large radii, small radii

## ## DETECTION

## ## ORBITAL PARAMETER FORWARD MODELING

## ## SENSITIVITY
