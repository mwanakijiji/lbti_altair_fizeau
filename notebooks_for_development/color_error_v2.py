# This calculates the difference in absolute magnitudes of a planet with and
# without atmospheric transmission correction. The purpose is to quantify the
# color error introduced by wavelength-variable atmospheric transmission

# Parent notebook created 2020 Apr. 30 by E.S.

# For the math, see research notebook fizeau_altair.tex on date 2020 Apr. 13
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
import os
import pysynphot as S

# # Needed ingredients (see notebook color_error_v2.ipynb for better readibility)

# ### The equation we want to solve is
# ### $M_{pl}^{(N)} - M_{pl}^{(calc)}
# =  2.5\textrm{log}_{10}	\left(
# 						\frac{
# 							\int d\lambda R_{\lambda} T_{\lambda} f_{\lambda,pl}^{(0)}
# 							}{
# 							\int d\lambda R_{\lambda} f_{\lambda,pl}^{(0)}
# 						}
# 						\frac{
# 							\int d\lambda R_{\lambda} f_{\lambda,Altair}^{(0)}
# 							}{
# 							\int d\lambda R_{\lambda} T_{\lambda} f_{\lambda,Altair}^{(0)}
# 							}
# 					\right)$
# ### or in terms of code variables,
# $M\_diff
# =  2.5\textrm{log}_{10} \left\{
# 	\frac{
# 	\textrm{piece1}
# 	}
# 	{
# 	\textrm{piece2}
# 	}
# 	\frac{
# 	\textrm{piece3}
# 	}{
# 	\textrm{piece4}
# 	}
# 	\right\}$
# ### where
# ### $M_{pl}^{(N)}$: absolute magnitude of a planet with NO atmospheric transmission effects
# ### $M_{pl}^{(T)}$ expected calculated absolute magnitude of a planet WITH atmospheric transmission effects
# ### $R_{\lambda}$: filter response curve
# ### $T_{\lambda}$: atmospheric transmission
# ### $f_{\lambda,Altair}^{(0)}$: flux of Altair at the top of the Earth's atmosphere
# ### $f_{\lambda,pl}^{(0)}$: flux of a planet at the top of the Earth's atmosphere

# ## Set some constants

# distance of Altair
d_altair_pc = 5.13 # units pc (and plus/minus 0.015 pc)

# angular and physical radius of Altair
solar_radii_per_altair_radii = 1.65 # units of solar radii (ref: Table 1 in Monnier+ 2007)

m_per_au = 1.4959787066e11 # units meters
m_per_solar_radius = 6.95508e8 # units meters
au_per_pc = np.divide(360*60*60,2*np.pi) # units AU/pc

# distance of Altair in Altair radii (this should be about 1.4e8 altair radii)
d_altair_altair_radii = d_altair_pc*\
                        au_per_pc*\
                        np.divide(1.,m_per_solar_radius)*\
                        m_per_au*\
                        np.divide(1.,solar_radii_per_altair_radii)

# zero point on the Vega scale, specific to Paranal-NACO NB405 filter, from SVO filter service
# http://svo2.cab.inta-csic.es/theory/fps/index.php?id=Paranal/NACO.NB405&&mode=browse&gname=Paranal&gname2=NACO#filter
zp_vega = 3.885e-12 # units erg /cm2 /sec /angstrom


# Naco NB405 filter
nb405_transmission = pd.read_csv("data/Paranal_NACO.NB405.dat.txt",
                                 names=["wavel_angs", "transmission"], delim_whitespace=True)

# plot filter
'''
plt.clf()
plt.scatter(nb405_transmission["wavel_angs"],nb405_transmission["transmission"])
plt.title("Paranal_NACO.NB405.dat.txt")
plt.xlabel("Wavelength ($\AA$)")
plt.ylabel("Transmission")
plt.savefig("junk.pdf")
plt.close()
'''

# ### Read in model spectra of planets

# Read in blackbodies in the general range of relevant temperatures of planets
# (note these are BB emissions at their surfaces)
# Ref: research journal, 2020 Mar. 4
# flux is in units of erg/cm2/s/A
bb_00200K = pd.read_csv("data/bb00200.dat.txt",
                                 names=["wavel_angs", "flux"], skiprows=6, delim_whitespace=True)
bb_00400K = pd.read_csv("data/bb00400.dat.txt",
                                 names=["wavel_angs", "flux"], skiprows=6, delim_whitespace=True)
bb_00650K = pd.read_csv("data/bb00650.dat.txt",
                                 names=["wavel_angs", "flux"], skiprows=6, delim_whitespace=True)
bb_02800K = pd.read_csv("data/bb02800.dat.txt",
                                 names=["wavel_angs", "flux"], skiprows=6, delim_whitespace=True)

# plot planet BBs
'''
plt.clf()
plt.plot(bb_00200K["wavel_angs"],np.divide(bb_00200K["flux"],np.max(bb_00200K["flux"])),
         label="BB, T=200K")
plt.plot(bb_00400K["wavel_angs"],np.divide(bb_00400K["flux"],np.max(bb_00400K["flux"])),
         label="BB, T=400K")
plt.plot(bb_00650K["wavel_angs"],np.divide(bb_00650K["flux"],np.max(bb_00650K["flux"])),
         label="BB, T=650K")
plt.plot(bb_02800K["wavel_angs"],np.divide(bb_02800K["flux"],np.max(bb_02800K["flux"])),
         label="BB, T=2800K")
plt.scatter(nb405_transmission["wavel_angs"],nb405_transmission["transmission"],s=2,
            label="NB405 trans.")
plt.title("Filter profile and planet BBs")
plt.xlabel("Wavelength (angstr)")
plt.ylabel("Normalized emission or filter transmission")
plt.xlim([0,1e5])
plt.legend()
plt.savefig("junk.pdf")
plt.close()
'''

# ### Obtain model spectra of host star (and of Vega, though we don't seem to need it)

# Spectrum meant to mimic Altair is from
# Kurucz ODFNEW /NOVER models
# teff = 7750 K (value for the effective temperature for the model. Temperatures are given in K)
# logg = 4.00 log(cm/s2) (value for Log(G) for the model.)
# meta = 0  (value for the Metallicity for the model.)
# lh = 1.25  (l/Hp where l is the  mixing length of the convective element and Hp is the pressure scale height)
# vtur = 2.0 km/s (Microturbulence velocity)
#
# column 1: WAVELENGTH (ANGSTROM), Wavelength in Angstrom
# column 2: FLUX (ERG/CM2/S/A), Flux in erg/cm2/s/A
model_spectrum = pd.read_csv("data/model_spec_teff_7750_logg_4_feh_0.txt",
                             names=["wavel_angs", "flux"], skiprows=9, delim_whitespace=True)

# plot host star model spectrum
'''
plt.clf()
plt.scatter(model_spectrum["wavel_angs"],model_spectrum["flux"])
plt.title("Model spectrum in the region around the filter bandpass")
plt.xlabel("Wavelength ($\AA$)")
plt.ylabel("Surface flux (erg/cm2/s/$\AA$)")
plt.ylim([0,1e5])
plt.xlim([38000,42000])
plt.savefig("junk.pdf")
plt.close()
'''

# Vega spectrum is imported from pysynphot
# refs:
# https://pysynphot.readthedocs.io
# https://www.stsci.edu/itt/review/synphot_2007/AppA_Catalogsa2.html
'''
print("Vega flux units are")
print(S.Vega.fluxunits.name)
'''
# flam = erg s−1 cm−2 \AA−1 (which is what I want! and at this amplitude it must be the flux at Earth)

# plot Vega spectrum
'''
plt.clf()
plt.plot(S.Vega.wave, S.Vega.flux)
plt.scatter(nb405_transmission["wavel_angs"],np.max(S.Vega.flux)*nb405_transmission["transmission"],s=2,
            label="NB405 trans.")
plt.xlim(0, 50000)
plt.xlabel(S.Vega.waveunits)
plt.ylabel(S.Vega.fluxunits)
plt.title(os.path.basename(S.Vega.name) + "\nand filter transmission")
plt.savefig("junk.pdf")
plt.close()
'''

# ### Read in atmospheric transmission
# source: https://atran.arc.nasa.gov/cgi-bin/atran/atran.cgi
# INPUT PARAMS for atran.plt.12172.dat:
'''
Altitude    :    10567.0000
 Water Vapor :    11.0000000
 Num layers  :            2
 Zenith Angle:    30.0000000
 Obs Lat     :    30.0000000
 Minimum Wave:    3.98000002
 Maximum Wave:    4.13000011
'''
trans_df = pd.read_csv("data/atran.plt.12172.dat", usecols=[1,2],
                       names=["wavel_um","transmission"], delim_whitespace=True)
# add column of wavelength in angstroms
trans_df["wavel_angs"] = (1e4)*trans_df["wavel_um"]

# plot
'''
plt.scatter(trans_df["wavel_angs"],trans_df["transmission"],s=2,label="55000* Atm. trans.")
plt.plot(nb405_transmission["wavel_angs"],
         np.divide(nb405_transmission["transmission"],np.max(nb405_transmission["transmission"])),label="NB405 trans.")
plt.title("Atmospheric transmission\n" + "Altitude    :    10567.0000 \n" + "Water Vapor :    11.0000000 \n" + "Num layers  :            2 \n" + "Zenith Angle:    30.0000000 \n" + "Obs Lat     :    30.0000000 \n" + "Minimum Wave:    3.98000002 \n" + "Maximum Wave:    4.13000011 \n")
plt.xlabel("Wavel (angstrom)")
plt.savefig("junk.pdf")
plt.close()
'''

# # Calculate pieces

# ## Interpolate all the curves (blackbodies, Altair, atmosphere) so that they
# ## are mapped onto the the abcissa of the filter transmission

# blackbody fluxes at the surface of the blackbody
bb_00200K_surf_flux_filter = np.interp(nb405_transmission["wavel_angs"].values,
                                             bb_00200K["wavel_angs"].values,
                                             bb_00200K["flux"].values)

bb_00400K_surf_flux_filter = np.interp(nb405_transmission["wavel_angs"].values,
                                             bb_00400K["wavel_angs"].values,
                                             bb_00400K["flux"].values)

bb_00650K_surf_flux_filter = np.interp(nb405_transmission["wavel_angs"].values,
                                             bb_00650K["wavel_angs"].values,
                                             bb_00650K["flux"].values)

bb_02800K_surf_flux_filter = np.interp(nb405_transmission["wavel_angs"].values,
                                             bb_02800K["wavel_angs"].values,
                                             bb_02800K["flux"].values)

# To do the integration over two functions represented by input arrays, use the abcissa
# of the filter transmission $R$ to make an interpolated form of $f$ pinned to the same
# ordinate. Then multiply the two arrays together and integrate over that.

altair_surf_flux_filter = np.interp(nb405_transmission["wavel_angs"].values,
                                             model_spectrum["wavel_angs"].values,
                                             model_spectrum["flux"].values)

vega_earth_flux_filter = np.interp(nb405_transmission["wavel_angs"].values,
                                             S.Vega.wave,
                                             S.Vega.flux)

# make smoothed version of the atmospheric transmission before interpolating
smoothed_atm_trans = scipy.signal.medfilt(trans_df["transmission"],kernel_size=401)
atm_transmission_filter = np.interp(nb405_transmission["wavel_angs"].values,
                                    trans_df["wavel_angs"],
                                    smoothed_atm_trans)

# plot an example interpolation
'''
plt.clf()
plt.scatter(bb_00200K["wavel_angs"].values,bb_00200K["flux"].values,label="original")
plt.plot(nb405_transmission["wavel_angs"].values,bb_00200K_surf_flux_filter,color="r",label="interpolated")
plt.legend()
plt.xlim([39000,42000]) # area of filter transmission
plt.savefig("junk.pdf")
plt.close()
'''

# ## Define the pieces

# ## piece1
# ### Integration of object flux, atmospheric transmission, and filter response (let's leave arguments $(\lambda)$ off for simplicity):
# ### $\int d\lambda R_{\lambda} T_{\lambda} f_{\lambda, obj}^{(0)} = \left( \frac{R_{obj}}{D} \right)^{2}\int d\lambda R_{\lambda} T_{\lambda} f_{\lambda, obj}^{(surf)} $
# ### However, the scaling factor $\left( \frac{R_{obj}}{D} \right)^{2}$ cancels in the fractions that go into the logarithm we want to calculate, so let's just calculate a quantity we call
# ## piece1_unscaled:
# ### $\int d\lambda R_{\lambda} T_{\lambda} f_{\lambda, obj}^{(surf)} $

# ## piece2
# ### Integration of object flux and filter response
# ### $\int d\lambda R_{\lambda} f_{\lambda, obj}^{(0)} = \left( \frac{R_{obj}}{D} \right)^{2}\int d\lambda R_{\lambda} f_{\lambda, obj}^{(surf)} $
# ### Again, removing the scaling factor which cancels anyway, we calculate
# ## piece2_unscaled:
# ### $\int d\lambda R_{\lambda} f_{\lambda, obj}^{(surf)} $

# ## piece3
# ### Integration of Altair flux and filter response
# ### $\int d\lambda R_{\lambda} f_{\lambda,Altair}^{(0)} = \left( \frac{R_{obj}}{D} \right)^{2}\int d\lambda R_{\lambda} f_{\lambda,Altair}^{(surf)} $
# ### Removing the scaling factor,
# ## piece3_unscaled:
# ### $\int d\lambda R_{\lambda} f_{\lambda,Altair}^{(surf)} $

# ## piece4
# ### Integration of Vega flux, atmospheric transmission, and filter response:
# ### $\int d\lambda R_{\lambda} T_{\lambda} f_{\lambda, Altair}^{(0)} = \left( \frac{R_{Altair}}{D} \right)^{2}\int d\lambda R_{\lambda} T_{\lambda} f_{\lambda,Altair}^{(surf)} $
# ### Removing the scaling factor,
# ## piece4_unscaled:
# ### $\int d\lambda R_{\lambda} T_{\lambda} f_{\lambda,Altair}^{(surf)} $

# ## Calculate the pieces

# calculate R*f integrands
integrand_unscaled_R_times_f_00200K = np.multiply(bb_00200K_surf_flux_filter,nb405_transmission["transmission"])
integrand_unscaled_R_times_f_00400K = np.multiply(bb_00400K_surf_flux_filter,nb405_transmission["transmission"])
integrand_unscaled_R_times_f_00650K = np.multiply(bb_00650K_surf_flux_filter,nb405_transmission["transmission"])
integrand_unscaled_R_times_f_02800K = np.multiply(bb_02800K_surf_flux_filter,nb405_transmission["transmission"])
integrand_unscaled_R_times_f_altair = np.multiply(altair_surf_flux_filter,nb405_transmission["transmission"])
# vega flux is already 'scaled' for distance because it's the value of the flux at Earth
integrand_scaled_R_times_flux_vega = np.multiply(vega_earth_flux_filter,nb405_transmission["transmission"])

# calculate R*f*lambda integrands
integrand_unscaled_R_times_f_times_lambda_00200K = np.multiply(nb405_transmission["wavel_angs"],integrand_unscaled_R_times_f_00200K)
integrand_unscaled_R_times_f_times_lambda_00400K = np.multiply(nb405_transmission["wavel_angs"],integrand_unscaled_R_times_f_00400K)
integrand_unscaled_R_times_f_times_lambda_00650K = np.multiply(nb405_transmission["wavel_angs"],integrand_unscaled_R_times_f_00650K)
integrand_unscaled_R_times_f_times_lambda_02800K = np.multiply(nb405_transmission["wavel_angs"],integrand_unscaled_R_times_f_02800K)
integrand_unscaled_R_times_f_times_lambda_altair = np.multiply(nb405_transmission["wavel_angs"],integrand_unscaled_R_times_f_altair)
# vega flux is already 'scaled' for distance because it's the value of the flux at Earth
integrand_scaled_R_times_flux_times_lambda_vega = np.multiply(nb405_transmission["wavel_angs"],integrand_scaled_R_times_flux_vega)

# calculate R*T*f integrands
integrand_unscaled_R_times_T_times_f_00200K = np.multiply(atm_transmission_filter,integrand_unscaled_R_times_f_00200K)
integrand_unscaled_R_times_T_times_f_00400K = np.multiply(atm_transmission_filter,integrand_unscaled_R_times_f_00400K)
integrand_unscaled_R_times_T_times_f_00650K = np.multiply(atm_transmission_filter,integrand_unscaled_R_times_f_00650K)
integrand_unscaled_R_times_T_times_f_02800K = np.multiply(atm_transmission_filter,integrand_unscaled_R_times_f_02800K)
integrand_unscaled_R_times_T_times_f_altair = np.multiply(atm_transmission_filter,integrand_unscaled_R_times_f_altair)
# vega flux is already 'scaled' for distance because it's the value of the flux at Earth
integrand_scaled_R_times_T_times_flux_vega = np.multiply(atm_transmission_filter,integrand_scaled_R_times_flux_vega)

# calculate R*T*f*lambda integrands
integrand_unscaled_R_times_T_times_f_times_lambda_00200K = np.multiply(nb405_transmission["wavel_angs"],integrand_unscaled_R_times_T_times_f_00200K)
integrand_unscaled_R_times_T_times_f_times_lambda_00400K = np.multiply(nb405_transmission["wavel_angs"],integrand_unscaled_R_times_T_times_f_00400K)
integrand_unscaled_R_times_T_times_f_times_lambda_00650K = np.multiply(nb405_transmission["wavel_angs"],integrand_unscaled_R_times_T_times_f_00650K)
integrand_unscaled_R_times_T_times_f_times_lambda_02800K = np.multiply(nb405_transmission["wavel_angs"],integrand_unscaled_R_times_T_times_f_02800K)
integrand_unscaled_R_times_T_times_f_times_lambda_altair = np.multiply(nb405_transmission["wavel_angs"],integrand_unscaled_R_times_T_times_f_altair)
# vega flux is already 'scaled' for distance because it's the value of the flux at Earth
integrand_scaled_R_times_T_times_flux_times_lambda_vega = np.multiply(nb405_transmission["wavel_angs"],integrand_scaled_R_times_T_times_flux_vega)


# plot R*f integrands

plt.clf()
plt.plot(nb405_transmission["wavel_angs"],nb405_transmission["transmission"],
        label="Straight transmission")
plt.plot(nb405_transmission["wavel_angs"],np.divide(integrand_unscaled_R_times_f_00200K,np.max(integrand_unscaled_R_times_f_00200K)),
         label="R * f(200K)")
plt.plot(nb405_transmission["wavel_angs"],np.divide(integrand_unscaled_R_times_f_00400K,np.max(integrand_unscaled_R_times_f_00400K)),
         label="R * f(400K)")
plt.plot(nb405_transmission["wavel_angs"],np.divide(integrand_unscaled_R_times_f_00650K,np.max(integrand_unscaled_R_times_f_00650K)),
         label="R * f(650K)")
plt.plot(nb405_transmission["wavel_angs"],np.divide(integrand_unscaled_R_times_f_02800K,np.max(integrand_unscaled_R_times_f_02800K)),
         label="R * f(2800K)")
plt.plot(nb405_transmission["wavel_angs"],np.divide(integrand_unscaled_R_times_f_altair,np.max(integrand_unscaled_R_times_f_altair)),
         label="R * f(Altair)")
plt.plot(nb405_transmission["wavel_angs"],np.divide(integrand_scaled_R_times_flux_vega,np.max(integrand_scaled_R_times_flux_vega)),
         label="R * f(Vega)")
plt.xlabel("Wavel (angstr)")
plt.legend()
plt.savefig("junk1.pdf")
plt.close()


# plot R*f*lambda integrands

plt.clf()
plt.plot(nb405_transmission["wavel_angs"],nb405_transmission["transmission"],
        label="Straight transmission")
plt.plot(nb405_transmission["wavel_angs"],np.divide(integrand_unscaled_R_times_f_times_lambda_00200K,np.max(integrand_unscaled_R_times_f_times_lambda_00200K)),
         label="R * lambda * f(200K)")
plt.plot(nb405_transmission["wavel_angs"],np.divide(integrand_unscaled_R_times_f_times_lambda_00400K,np.max(integrand_unscaled_R_times_f_times_lambda_00400K)),
         label="R * lambda * f(400K)")
plt.plot(nb405_transmission["wavel_angs"],np.divide(integrand_unscaled_R_times_f_times_lambda_00650K,np.max(integrand_unscaled_R_times_f_times_lambda_00650K)),
         label="R * lambda * f(650K)")
plt.plot(nb405_transmission["wavel_angs"],np.divide(integrand_unscaled_R_times_f_times_lambda_02800K,np.max(integrand_unscaled_R_times_f_times_lambda_02800K)),
         label="R * lambda * f(2800K)")
plt.plot(nb405_transmission["wavel_angs"],np.divide(integrand_unscaled_R_times_f_times_lambda_altair,np.max(integrand_unscaled_R_times_f_times_lambda_altair)),
         label="R * lambda * f(Altair)")
plt.plot(nb405_transmission["wavel_angs"],np.divide(integrand_scaled_R_times_flux_times_lambda_vega,np.max(integrand_scaled_R_times_flux_times_lambda_vega)),
         label="R * lambda * f(Vega)")
plt.xlabel("Wavel (angstr)")
plt.legend()
plt.savefig("junk2.pdf")
plt.close()


# plot R*T*f integrands
plt.clf()
plt.plot(nb405_transmission["wavel_angs"],nb405_transmission["transmission"], label="Straight filter transmission")
plt.plot(nb405_transmission["wavel_angs"],atm_transmission_filter, label="Atmospheric transmission")
plt.plot(nb405_transmission["wavel_angs"],np.divide(integrand_unscaled_R_times_T_times_f_00200K,np.max(integrand_unscaled_R_times_T_times_f_00200K)),
         label="R * T * f(200K)")
plt.plot(nb405_transmission["wavel_angs"],np.divide(integrand_unscaled_R_times_T_times_f_00400K,np.max(integrand_unscaled_R_times_T_times_f_00400K)),
         label="R * T * f(400K)")
plt.plot(nb405_transmission["wavel_angs"],np.divide(integrand_unscaled_R_times_T_times_f_00650K,np.max(integrand_unscaled_R_times_T_times_f_00650K)),
         label="R * T * f(650K)")
plt.plot(nb405_transmission["wavel_angs"],np.divide(integrand_unscaled_R_times_T_times_f_02800K,np.max(integrand_unscaled_R_times_T_times_f_02800K)),
         label="R * T * f(2800K)")
plt.plot(nb405_transmission["wavel_angs"],np.divide(integrand_unscaled_R_times_T_times_f_altair,np.max(integrand_unscaled_R_times_T_times_f_altair)),
         label="R * T * f(Altair)")
plt.plot(nb405_transmission["wavel_angs"],np.divide(integrand_scaled_R_times_T_times_flux_vega,np.max(integrand_scaled_R_times_T_times_flux_vega)),
         label="R * T * f(Vega)")
plt.xlabel("Wavel (angstr)")
plt.legend()
plt.savefig("junk3.pdf")
plt.close()


# plot R*T*f*lambda integrands

plt.clf()
plt.plot(nb405_transmission["wavel_angs"],nb405_transmission["transmission"], label="Straight filter transmission")
plt.plot(nb405_transmission["wavel_angs"],atm_transmission_filter, label="Atmospheric transmission")
plt.plot(nb405_transmission["wavel_angs"],np.divide(integrand_unscaled_R_times_T_times_f_times_lambda_00200K,np.max(integrand_unscaled_R_times_T_times_f_times_lambda_00200K)),
         label="R * T * lamdba * f(200K)")
plt.plot(nb405_transmission["wavel_angs"],np.divide(integrand_unscaled_R_times_T_times_f_times_lambda_00400K,np.max(integrand_unscaled_R_times_T_times_f_times_lambda_00400K)),
         label="R * T * lamdba * f(400K)")
plt.plot(nb405_transmission["wavel_angs"],np.divide(integrand_unscaled_R_times_T_times_f_times_lambda_00650K,np.max(integrand_unscaled_R_times_T_times_f_times_lambda_00650K)),
         label="R * T * lamdba * f(650K)")
plt.plot(nb405_transmission["wavel_angs"],np.divide(integrand_unscaled_R_times_T_times_f_times_lambda_02800K,np.max(integrand_unscaled_R_times_T_times_f_times_lambda_02800K)),
         label="R * T * lamdba * f(2800K)")
plt.plot(nb405_transmission["wavel_angs"],np.divide(integrand_unscaled_R_times_T_times_f_times_lambda_altair,np.max(integrand_unscaled_R_times_T_times_f_times_lambda_altair)),
         label="R * T * lamdba * f(Altair)")
plt.plot(nb405_transmission["wavel_angs"],np.divide(integrand_scaled_R_times_T_times_flux_times_lambda_vega,np.max(integrand_scaled_R_times_T_times_flux_times_lambda_vega)),
         label="R * T * lamdba * f(Vega)")
plt.xlabel("Wavel (angstr)")
plt.legend()
plt.savefig("junk4.pdf")
plt.close()


# plot a couple by themselves without normalization, so that it's checkable
'''
plt.clf()
plt.plot(nb405_transmission["wavel_angs"],integrand_unscaled_R_times_T_times_f_altair,
         label="R*T*Altair")
plt.xlabel("Wavel (angstr)")
plt.legend()
plt.savefig("junk5.pdf")
plt.close()
plt.clf()
plt.plot(nb405_transmission["wavel_angs"],integrand_unscaled_R_times_f_02800K,
         label="R*f_2800K")
plt.xlabel("Wavel (angstr)")
plt.legend()
plt.savefig("junk6.pdf")
plt.close()
'''

# integrate (for energy-counters, without the extra lambda in the integrands)
piece_1_unscaled_00200K = np.trapz(integrand_unscaled_R_times_T_times_f_00200K,x=nb405_transmission["wavel_angs"])
piece_1_unscaled_00400K = np.trapz(integrand_unscaled_R_times_T_times_f_00400K,x=nb405_transmission["wavel_angs"])
piece_1_unscaled_00650K = np.trapz(integrand_unscaled_R_times_T_times_f_00650K,x=nb405_transmission["wavel_angs"])
piece_1_unscaled_02800K = np.trapz(integrand_unscaled_R_times_T_times_f_02800K,x=nb405_transmission["wavel_angs"])

piece_2_unscaled_00200K = np.trapz(integrand_unscaled_R_times_f_00200K,x=nb405_transmission["wavel_angs"])
piece_2_unscaled_00400K = np.trapz(integrand_unscaled_R_times_f_00400K,x=nb405_transmission["wavel_angs"])
piece_2_unscaled_00650K = np.trapz(integrand_unscaled_R_times_f_00650K,x=nb405_transmission["wavel_angs"])
piece_2_unscaled_02800K = np.trapz(integrand_unscaled_R_times_f_02800K,x=nb405_transmission["wavel_angs"])

piece_3_unscaled_altair = np.trapz(integrand_unscaled_R_times_f_altair,x=nb405_transmission["wavel_angs"])
piece_4_unscaled_altair = np.trapz(integrand_unscaled_R_times_T_times_f_altair,x=nb405_transmission["wavel_angs"])

piece_3_scaled_vega = np.trapz(integrand_scaled_R_times_flux_vega,x=nb405_transmission["wavel_angs"])
piece_4_scaled_vega = np.trapz(integrand_scaled_R_times_T_times_flux_vega,x=nb405_transmission["wavel_angs"])

# integrate (for photon-counters, with the extra lambda in the integrands)
piece_1_unscaled_00200K_lambda = np.trapz(integrand_unscaled_R_times_T_times_f_times_lambda_00200K,x=nb405_transmission["wavel_angs"])
piece_1_unscaled_00400K_lambda = np.trapz(integrand_unscaled_R_times_T_times_f_times_lambda_00400K,x=nb405_transmission["wavel_angs"])
piece_1_unscaled_00650K_lambda = np.trapz(integrand_unscaled_R_times_T_times_f_times_lambda_00650K,x=nb405_transmission["wavel_angs"])
piece_1_unscaled_02800K_lambda = np.trapz(integrand_unscaled_R_times_T_times_f_times_lambda_02800K,x=nb405_transmission["wavel_angs"])

piece_2_unscaled_00200K_lambda = np.trapz(integrand_unscaled_R_times_f_times_lambda_00200K,x=nb405_transmission["wavel_angs"])
piece_2_unscaled_00400K_lambda = np.trapz(integrand_unscaled_R_times_f_times_lambda_00400K,x=nb405_transmission["wavel_angs"])
piece_2_unscaled_00650K_lambda = np.trapz(integrand_unscaled_R_times_f_times_lambda_00650K,x=nb405_transmission["wavel_angs"])
piece_2_unscaled_02800K_lambda = np.trapz(integrand_unscaled_R_times_f_times_lambda_02800K,x=nb405_transmission["wavel_angs"])

piece_3_unscaled_altair_lambda = np.trapz(integrand_unscaled_R_times_f_times_lambda_altair,x=nb405_transmission["wavel_angs"])
piece_4_unscaled_altair_lambda = np.trapz(integrand_unscaled_R_times_T_times_f_times_lambda_altair,x=nb405_transmission["wavel_angs"])

piece_3_scaled_vega_lambda = np.trapz(integrand_scaled_R_times_flux_times_lambda_vega,x=nb405_transmission["wavel_angs"])
piece_4_scaled_vega_lambda = np.trapz(integrand_scaled_R_times_T_times_flux_times_lambda_vega,x=nb405_transmission["wavel_angs"])

# # Put pieces together

# ### As a reminder, the equation we want to solve is
# ### $M_{pl}^{(N)} - M_{pl}^{(calc)}
# =  2.5\textrm{log}_{10}	\left(
# 						\frac{
# 							\int d\lambda R_{\lambda} T_{\lambda} f_{\lambda,pl}^{(0)}
# 							}{
# 							\int d\lambda R_{\lambda} f_{\lambda,pl}^{(0)}
# 						}
# 						\frac{
# 							\int d\lambda R_{\lambda} f_{\lambda,star}^{(0)}
# 							}{
# 							\int d\lambda R_{\lambda} T_{\lambda} f_{\lambda,star}^{(0)}
# 							}
# 					\right)$
# ### or in terms of code variables,
# $M\_diff
# =  2.5\textrm{log}_{10} \left\{
# 	\frac{
# 	\textrm{piece1}
# 	}
# 	{
# 	\textrm{piece2}
# 	}
# 	\frac{
# 	\textrm{piece3}
# 	}{
# 	\textrm{piece4}
# 	}
# 	\right\}$

# a plot of many things for the Altair paper integrand
#print(np.max())
'''
plt.clf()
#color = 'tab:red'
plt.xlabel('Wavelength ($\AA$)')
plt.ylabel('Transmission')
plt.plot(nb405_transmission["wavel_angs"],nb405_transmission["transmission"],
         label="NACO NB4.05 filter ($R_{\lambda}$)")
plt.plot(trans_df["wavel_angs"],trans_df["transmission"],
         label="Atmosphere (high-res)")
plt.plot(nb405_transmission["wavel_angs"].values, atm_transmission_filter,
         label="Atmosphere ($T_{\lambda}$)")
plt.xlim([39800,41300])
plt.legend(loc="upper right")
plt.show()
'''

# FINAL ANSWERS (energy counters)
M_diff_00200K = 2.5*np.log10(np.divide(piece_1_unscaled_00200K,piece_2_unscaled_00200K)*
                         np.divide(piece_3_unscaled_altair,piece_4_unscaled_altair))
M_diff_00400K = 2.5*np.log10(np.divide(piece_1_unscaled_00400K,piece_2_unscaled_00400K)*
                         np.divide(piece_3_unscaled_altair,piece_4_unscaled_altair))
M_diff_00650K = 2.5*np.log10(np.divide(piece_1_unscaled_00650K,piece_2_unscaled_00650K)*
                         np.divide(piece_3_unscaled_altair,piece_4_unscaled_altair))
M_diff_02800K = 2.5*np.log10(np.divide(piece_1_unscaled_02800K,piece_2_unscaled_02800K)*
                         np.divide(piece_3_unscaled_altair,piece_4_unscaled_altair))
print("M_diff_00200K (energy counter):")
print(M_diff_00200K)
print("M_diff_00400K (energy counter):")
print(M_diff_00400K)
print("M_diff_00650K (energy counter):")
print(M_diff_00650K)
print("M_diff_02800K (energy counter):")
print(M_diff_02800K)

# FINAL ANSWERS (photon counters)
M_diff_00200K_lambda = 2.5*np.log10(np.divide(piece_1_unscaled_00200K_lambda,piece_2_unscaled_00200K_lambda)*
                         np.divide(piece_3_unscaled_altair_lambda,piece_4_unscaled_altair_lambda))
M_diff_00400K_lambda = 2.5*np.log10(np.divide(piece_1_unscaled_00400K_lambda,piece_2_unscaled_00400K_lambda)*
                         np.divide(piece_3_unscaled_altair_lambda,piece_4_unscaled_altair_lambda))
M_diff_00650K_lambda = 2.5*np.log10(np.divide(piece_1_unscaled_00650K_lambda,piece_2_unscaled_00650K_lambda)*
                         np.divide(piece_3_unscaled_altair_lambda,piece_4_unscaled_altair_lambda))
M_diff_02800K_lambda = 2.5*np.log10(np.divide(piece_1_unscaled_02800K_lambda,piece_2_unscaled_02800K_lambda)*
                         np.divide(piece_3_unscaled_altair_lambda,piece_4_unscaled_altair_lambda))
print("M_diff_00200K (photon counter):")
print(M_diff_00200K_lambda)
print("M_diff_00400K (photon counter):")
print(M_diff_00400K_lambda)
print("M_diff_00650K (photon counter):")
print(M_diff_00650K_lambda)
print("M_diff_02800K (photon counter):")
print(M_diff_02800K_lambda)