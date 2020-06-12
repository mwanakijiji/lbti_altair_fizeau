# This calculate the absolute magnitude of Altair, so as to determine
# absolute magnitudes of fake companions for making a contrast curve

# Notebook parent created 2020 Feb. 27 by E.S.

# For the math, see research notebook fizeau_altair.tex on date 2020 Feb. 27

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pysynphot as S
import scipy
import os


# ### Set some constants

# distance of Altair
d_altair_pc = 5.13 # units pc (and plus/minus 0.015 pc)

# angular and physical radius of Altair
solar_radii_per_altair_radii = 1.65 # units of solar radii (ref: Table 1 in Monnier+ 2007)

# other relations
m_per_au = 1.4959787066e11 # units meters/AU
m_per_solar_radius = 6.95508e8 # units meters/R_sol
au_per_pc = np.divide(360*60*60,2*np.pi) # units AU/pc

# distance of Altair in Altair radii (this should be about 1.4e8 altair radii)
d_altair_altair_radii = d_altair_pc*au_per_pc*np.divide(1.,m_per_solar_radius)*\
m_per_au*np.divide(1.,solar_radii_per_altair_radii)

'''
# plot filter

plt.clf()
plt.scatter(filter_transmission["wavel_angs"],filter_transmission["transmission"])
plt.title("Paranal_NACO.NB405.dat.txt")
plt.xlabel("Wavelength ($\AA$)")
plt.ylabel("Transmission")
plt.savefig("junk1.pdf")
plt.close()
'''

def altair_abs_mag(filter,zp):
    '''
    Determine absolute magnitude of Altair through different filters: a few
    for consistency checks, and the NB4.05 one for our science

    INPUTS:
    filter: string to set the filter to read in
        "john_U"/"2mass_j"/"2mass_h"/"2mass_ks"
    zp: zero point on Vega scale to use
    '''

    # ### Read in filter curve: NACO NB4.05 to approximate LMIRcam Br-alpha
    if (filter == "2mass_j"):
        filter_transmission = pd.read_csv("./modules/data/2MASS_J_dat.txt",
                                     names=["wavel_angs", "transmission"], delim_whitespace=True)
        info_string1 = "Absolute M_star in 2MASS J filter on Vega scale:"
        true_m_rel = 0.313
        info_string2 = "Relative m_star in 2MASS J filter on Vega scale (true answer m_rel in 2MASS J: "+str(true_m_rel)+"):"
    elif (filter == "2mass_h"):
        filter_transmission = pd.read_csv("./modules/data/2MASS_H_dat.txt",
                                     names=["wavel_angs", "transmission"], delim_whitespace=True)
        info_string1 = "Absolute M_star in 2MASS H filter on Vega scale:"
        true_m_rel = 0.102
        info_string2 = "Relative m_star in 2MASS H filter on Vega scale (true answer m_rel in 2MASS H: "+str(true_m_rel)+"):"
    elif (filter == "2mass_ks"):
        filter_transmission = pd.read_csv("./modules/data/2MASS_Ks_dat.txt",
                                     names=["wavel_angs", "transmission"], delim_whitespace=True)
        info_string1 = "Absolute M_star in 2MASS Ks filter on Vega scale:"
        true_m_rel = 0.102
        info_string2 = "Relative m_star in 2MASS Ks filter on Vega scale (true answer m_rel in 2MASS K: "+str(true_m_rel)+"):"
    elif (filter == "john_U"):
        filter_transmission = pd.read_csv("./modules/data/Generic_Johnson_UBVRIJHKL.U.dat.txt",
                                     names=["wavel_angs", "transmission"], delim_whitespace=True)
        info_string1 = "Absolute M_star in Johnson U filter on Vega scale:"
        true_m_rel = 1.07
        info_string2 = "Relative m_star in Johnson U filter on Vega scale (true answer m_rel in Johnson U: "+str(true_m_rel)+"):"
    elif (filter == "john_B"):
        filter_transmission = pd.read_csv("./modules/data/Generic_Johnson_UBVRIJHKL.B.dat.txt",
                                     names=["wavel_angs", "transmission"], delim_whitespace=True)
        info_string1 = "Absolute M_star in Johnson B filter on Vega scale:"
        true_m_rel = 0.98
        info_string2 = "Relative m_star in Johnson B filter on Vega scale (true answer m_rel in Johnson B: "+str(true_m_rel)+"):"
    elif (filter == "john_V"):
        filter_transmission = pd.read_csv("./modules/data/Generic_Johnson_UBVRIJHKL.V.dat.txt",
                                     names=["wavel_angs", "transmission"], delim_whitespace=True)
        info_string1 = "Absolute M_star in Johnson V filter on Vega scale:"
        true_m_rel = 0.76
        info_string2 = "Relative m_star in Johnson V filter on Vega scale (true answer m_rel in Johnson V: "+str(true_m_rel)+"):"
    elif (filter == "john_R"):
        filter_transmission = pd.read_csv("./modules/data/Generic_Johnson_UBVRIJHKL.R.dat.txt",
                                     names=["wavel_angs", "transmission"], delim_whitespace=True)
        info_string1 = "Absolute M_star in Johnson R filter on Vega scale:"
        true_m_rel = 0.62
        info_string2 = "Relative m_star in Johnson R filter on Vega scale (true answer m_rel in Johnson R: "+str(true_m_rel)+"):"
    elif (filter == "john_I"):
        filter_transmission = pd.read_csv("./modules/data/Generic_Johnson_UBVRIJHKL.I.dat.txt",
                                     names=["wavel_angs", "transmission"], delim_whitespace=True)
        info_string1 = "Absolute M_star in Johnson I filter on Vega scale:"
        true_m_rel = 0.49
        info_string2 = "Relative m_star in Johnson I filter on Vega scale (true answer m_rel in Johnson I: "+str(true_m_rel)+"):"
    elif (filter == "john_J"):
        filter_transmission = pd.read_csv("./modules/data/Generic_Johnson_UBVRIJHKL.J.dat.txt",
                                     names=["wavel_angs", "transmission"], delim_whitespace=True)
        info_string1 = "Absolute M_star in Johnson J filter on Vega scale:"
        true_m_rel = 0.35
        info_string2 = "Relative m_star in Johnson J filter on Vega scale (true answer m_rel in Johnson J: "+str(true_m_rel)+"):"
    elif (filter == "john_K"):
        filter_transmission = pd.read_csv("./modules/data/Generic_Johnson_UBVRIJHKL.K.dat.txt",
                                     names=["wavel_angs", "transmission"], delim_whitespace=True)
        info_string1 = "Absolute M_star in Johnson K filter on Vega scale:"
        true_m_rel = 0.24
        info_string2 = "Relative m_star in Johnson K filter on Vega scale (true answer m_rel in Johnson K: "+str(true_m_rel)+"):"
    elif (filter == "nb405"):
        filter_transmission = pd.read_csv("./modules/data/Paranal_NACO.NB405.dat.txt",
                                     names=["wavel_angs", "transmission"], delim_whitespace=True)
        info_string1 = "Absolute M_star in NACO NB4.05 filter on Vega scale:"
        true_m_rel = np.nan
        info_string2 = ""

    # ### Read in model spectra (of a host star and Vega, though we likely
    # ### don't need the host star here)


    ##################################################################################################
    # Surface flux of a model spectrum, based on a Kurucz model, courtesy SVO
    # I tried to approximate parameters of Altair, which are
    # Teff=7550 K, logg=4.13, [Fe/H]=-0.24 \citep{erspamer2003automated}
    # Kurucz ODFNEW /NOVER models
    # teff = 7750 K (value for the effective temperature for the model. Temperatures are given in K)
    # logg = 4.00 log(cm/s2) (value for Log(G) for the model.)
    # meta = 0  (value for the Metallicity for the model.)
    # lh = 1.25  (l/Hp where l is the  mixing length of the convective element and Hp is the pressure scale height)
    # vtur = 2.0 km/s (Microturbulence velocity)
    #
    # column 1: WAVELENGTH (ANGSTROM), Wavelength in Angstrom
    # column 2: FLUX (ERG/CM2/S/A), Flux in erg/cm2/s/A
    model_spectrum = pd.read_csv("./modules/data/model_spec_teff_7750_logg_4_feh_0.txt",
                                 names=["wavel_angs", "flux"], skiprows=9, delim_whitespace=True)

    # plot spectrum
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
    # flam = erg s−1 cm−2 \AA−1 (which is what I want! and at this amplitude it
    # must be the flux at Earth, not the surface flux)

    '''
    plt.clf()
    plt.plot(S.Vega.wave, S.Vega.flux)
    plt.scatter(filter_transmission["wavel_angs"],np.max(S.Vega.flux)*filter_transmission["transmission"],s=2,
                label="NB405 trans.")
    plt.xlim(0, 50000)
    plt.xlabel(S.Vega.waveunits)
    plt.ylabel(S.Vega.fluxunits)
    plt.title(os.path.basename(S.Vega.name) + "\nand filter transmission")
    plt.savefig("junk.pdf")
    plt.close()
    '''

    # ## Figure out what we want (see notebook for more readable equations)

    # ### The equation for absolute magnitude (on the Vega scale) of a star we want to solve is
    # ### $M_{ \textrm{star}} = m_{ \textrm{star}}  - 5\textrm{log}_{10}\left( \frac{\textrm{d}}{\textrm{10 pc}} \right)$
    # ### where
    # ### $m_{ \textrm{star}} = -2.5 \textrm{log}_{10} \left\{ \frac{\int d\lambda R_{\lambda}(\lambda) f_{\lambda}^{(0, star)}(\lambda) / \int d\lambda R_{\lambda}(\lambda) }{\textrm{zp}(f_{\lambda}^{(0, Vega)})}  \right\}$

    # ### The equation for absolute magnitude (on the Vega scale) of a star we want to solve is
    # ### $M_{ \textrm{star}} = m_{ \textrm{star}}  - 5\textrm{log}_{10}\left( \frac{\textrm{d}}{\textrm{10 pc}} \right)$
    # ### where
    # ### $m_{ \textrm{star}} = -2.5 \textrm{log}_{10} \left\{ \frac{\int d\lambda R_{\lambda}(\lambda) f_{\lambda,star}^{(0)}(\lambda)}{\int d\lambda R_{\lambda}(\lambda) }\right\} + 2.5 \textrm{log}_{10} \left\{ \textrm{zp}(f_{\lambda,Vega}^{(0)}) \right\}= -2.5 \textrm{log}_{10} \left\{ \frac{\int d\lambda R_{\lambda}(\lambda) f_{\lambda,star}^{(0)}(\lambda) / \int d\lambda R_{\lambda}(\lambda) }{\textrm{zp}(f_{\lambda, Vega}^{(0)})}  \right\}$
    # ### The $(0)$ indicates 'at the top of Earth's atmosphere' and the zero point is defined on SVO page as
    # ### $\textrm{zp}(f_{\lambda,Vega}^{(0)})\equiv \frac{ \int d\lambda R(\lambda)f_{\lambda,Vega}^{(0)}(\lambda)}{\int d\lambda R(\lambda)}$
    # ### To find the expected flux of the star at the top of the Earth's atmosphere, use a synthetic spectrum (we are in the Rayleigh-Jeans regime for an Altair-like spectrum anyway) and scale it for distance:
    # ### $f_{\lambda, star}^{(0)}(\lambda) = \left( \frac{R_{star}}{D} \right)^{2} f_{\lambda,star}^{(surf)}(\lambda) $
    # ### Putting all the pieces together in terms of fundamental measurables, we want to calculate
    # ### $M_{ \textrm{star}} = -2.5 \textrm{log}_{10} \left\{ \left( \frac{R_{star}}{D} \right)^{2}\frac{\int d\lambda R_{\lambda}(\lambda) f_{\lambda,star}^{(surf)}(\lambda) / \int d\lambda R_{\lambda}(\lambda) }{\textrm{zp}(f_{\lambda, Vega}^{(0)})}  \right\}   - 5\textrm{log}_{10}\left( \frac{\textrm{d}}{\textrm{10 pc}} \right)$
    # ### or in code variables,
    # ### $M_{ \textrm{star}} = -2.5 \textrm{log}_{10} \left\{ \textrm{piece_A}\frac{\textrm{piece_B} / \textrm{piece_C} }{\textrm{zp_vega}} \right\}   - 5\textrm{log}_{10}\left( \textrm{piece_D} \right)$

    # ## Calculate the pieces

    # ### $\textrm{piece_A}$: Calculate inverse-square law scaling factor to convert surface flux $f_{\lambda}^{(surf, star)}(\lambda)$ to flux at top of Earth's atmosphere:
    # ### $\textrm{piece_A} \equiv \left( \frac{R_{star}}{D} \right)^{2} $
    piece_A = np.power(np.divide(1.,d_altair_altair_radii),2.)

    # ### $\textrm{piece_B}$: Integration of star flux and filter response:
    # ### $\textrm{piece_B} \equiv \int d\lambda R_{\lambda}(\lambda) f_{\lambda,star}^{(surf)}(\lambda) $

    # To do the integration over two functions represented by input arrays, use the abcissa
    # of the filter transmission $R$ to make an interpolated form of $f$ pinned to the same
    # ordinate. Then multiply the two arrays together and integrate over that.
    model_surf_flux_filter_abcissa = np.interp(filter_transmission["wavel_angs"].values,
                                                 model_spectrum["wavel_angs"].values,
                                                 model_spectrum["flux"].values)

    # now multiply the surface flux by the filter transmission to obtain the integrand
    integrand_piece_B = np.multiply(filter_transmission["transmission"],
                                    model_surf_flux_filter_abcissa)
    # integrate
    piece_B = np.trapz(integrand_piece_B,x=filter_transmission["wavel_angs"])

    # ### $\textrm{piece_C}$: The normalization constant:
    # ### $\textrm{piece_C} \equiv \int d\lambda R_{\lambda}(\lambda) $
    # integrate the filter transmission
    piece_C = np.trapz(filter_transmission["transmission"],x=filter_transmission["wavel_angs"])

    # ### $\textrm{piece_D}$: Multiples of 10 pc:
    # ### $\textrm{piece_D} \equiv \frac{\textrm{5.13 pc}}{\textrm{10 pc}} $
    piece_D = np.divide(5.13,10)

    # ### Also check the zero point given by SVO:
    # ### $\textrm{zp_vega} \equiv \frac{\int d\lambda R_{\lambda} f_{Vega}^{(0)}}{\int d\lambda R_{\lambda}} \equiv \frac{\textrm{piece_vega}}{\textrm{piece_C}} $
    # ### SVO value is 3.885e-12	(erg/cm2/s/A)
    '''
    ## as SVO defines it
    # interpolate to put Vega flux on same abcissa
    vega_earth_flux_filter_abcissa = np.interp(filter_transmission["wavel_angs"].values,
                                               S.Vega.wave,
                                               S.Vega.flux)
    # now multiply the surface flux by the filter transmission to obtain the integrand
    integrand_piece_vega = np.multiply(filter_transmission["transmission"],vega_earth_flux_filter_abcissa)
    # integrate
    piece_vega = np.trapz(integrand_piece_vega,x=filter_transmission["wavel_angs"])
    zp_vega_svo_defined_calc = np.divide(piece_vega,piece_C)
    print("zp_vega_svo_defined_calc:")
    print(zp_vega_svo_defined_calc)


    # ### check zero point value using other definition, with lambda inside integrand (which seems to appear in other Vega zero points):
    #
    # ### $\textrm{zp_vega_extra_lambd} \equiv \frac{\int d\lambda R_{\lambda} \lambda f_{Vega}^{(0)}}{\int d\lambda R_{\lambda} \lambda } \equiv \frac{\textrm{piece_vega_extra_lambd}}{\textrm{piece_C_extra_lambd}} $
    # now multiply the surface flux by the filter transmission to obtain the integrand
    integrand_piece_vega_extra_lambd = np.multiply(integrand_piece_vega,filter_transmission["wavel_angs"])
    # integrate to get numerator
    piece_vega_extra_lambd = np.trapz(integrand_piece_vega_extra_lambd,x=filter_transmission["wavel_angs"])
    # integrand of denominator
    integrand_piece_C_extra_lambd = np.multiply(filter_transmission["transmission"],filter_transmission["wavel_angs"])
    # integrate to get denominator
    piece_C_extra_lambd = np.trapz(integrand_piece_C_extra_lambd,x=filter_transmission["wavel_angs"])
    zp_vega_extra_lambda_calc = np.divide(piece_vega_extra_lambd,piece_C_extra_lambd)
    print("zp_vega_extra_lambda_calc:")
    print(zp_vega_extra_lambda_calc)
    # ### ... so the zero points are virtually the same
    '''

    # ### Put everything together to get an apparent magnitude. Reiterating, we want
    # ### $M_{ \textrm{star}} = -2.5 \textrm{log}_{10} \left\{ \left( \frac{R_{star}}{D} \right)^{2}\frac{\int d\lambda R_{\lambda}(\lambda) f_{\lambda,star}^{(surf)}(\lambda) / \int d\lambda R_{\lambda}(\lambda) }{\textrm{zp}(f_{\lambda, Vega}^{(0)})}  \right\}   - 5\textrm{log}_{10}\left( \frac{\textrm{d}}{\textrm{10 pc}} \right)$
    # ### $\equiv -2.5 \textrm{log}_{10} \left\{ \textrm{piece_A}\frac{\textrm{piece_B} / \textrm{piece_C} }{\textrm{zp_vega}} \right\}   - 5\textrm{log}_{10}\left( \textrm{piece_D} \right)$

    # ## Put pieces together for finding M_star

    print("piece_A:")
    print(piece_A)
    print("piece_B:")
    print(piece_B)
    print("piece_C:")
    print(piece_C)
    print("zp_vega:")
    print(zp)
    print("piece_D:")
    print(piece_D)

    # ### Final answer
    M_star =    -2.5*np.log10(piece_A*np.divide(
                                    np.divide(piece_B,piece_C),
                                    zp))\
                -5*np.log10(piece_D)
    print("---------------")
    print(info_string1)
    print(M_star)
    print("---------------")

    # Use distance modulus to find apparent magnitude
    m_rel_star = M_star + 5.*np.log10(d_altair_pc/10.)
    print("Relative m_star calculated by us:")
    print(m_rel_star)
    print(info_string2)
    print("Difference with measured elsewhere (m_star - m_star_measured):")
    print(np.round(np.subtract(m_rel_star,true_m_rel),3))
    print("---------------")
    # ### Note 2MASS measured $m_{K}=0.0102$.
    # ### Then, $M_{K}=m_{K}-5\textrm{log}_{10}\left(\frac{\textrm{5.13 pc}}{\textrm{10 pc}}\right) = 1.55$

    # make an FYI plot everything so far around the filter region
    '''
    plt.clf()
    plt.scatter(filter_transmission["wavel_angs"],
                np.divide(model_surf_flux_filter_abcissa,np.max(model_surf_flux_filter_abcissa)),
                label="Model spectrum f")
    plt.scatter(filter_transmission["wavel_angs"],
                np.divide(filter_transmission["transmission"],np.max(filter_transmission["transmission"])),
                label="Filter response R")
    plt.scatter(filter_transmission["wavel_angs"],
                np.divide(integrand_piece_B,np.max(integrand_piece_B)),
                label="R*f")
    plt.title("Filter " + filter +
                "\nm_star calculated by us: " + str(np.round(m_rel_star,3)) +
                "\nm_star measured elsewhere: " + str(np.round(true_m_rel,3)))
    plt.legend()
    plt.xlabel("Wavel ($\AA$)")
    plt.ylabel("Normalized quantities")
    plt.savefig("junk_filter_"+filter+"_zoom.pdf")
    plt.close()
    '''

    # make a more global plot of the model spectrum
    if (filter=="john_U"): # if statement here to allow overplotting outside function
        plt.figure(figsize=(16,8))
    if ("john" in filter):
        plt.plot(np.divide(filter_transmission["wavel_angs"],1e4),
                np.divide(filter_transmission["transmission"],np.max(filter_transmission["transmission"])),
                color="darkgreen", alpha=0.5, linewidth=4,
                label=filter)
    if ("mass" in filter):
        plt.plot(np.divide(filter_transmission["wavel_angs"],1e4),
                np.divide(filter_transmission["transmission"],np.max(filter_transmission["transmission"])),
                color="red", alpha=0.5, linewidth=4,
                label=filter)
    if (filter=="nb405"):
        plt.plot(np.divide(filter_transmission["wavel_angs"],1e4),
                np.divide(filter_transmission["transmission"],np.max(filter_transmission["transmission"])),
                color="orange", linewidth=4,
                label=filter)
        plt.plot(np.divide(model_spectrum["wavel_angs"],1e4),
                np.divide(model_spectrum["flux"],np.max(model_spectrum["flux"])),color="k", linewidth=4,
                label="Altair model")
        plt.ylim([0,1.1])
        plt.xlim([0,5])
        #plt.yscale('log')
        #plt.title("Filter " + filter +
        #            "\nm_star calculated by us: " + str(np.round(m_rel_star,3)) +
        #            "\nm_star measured elsewhere: " + str(np.round(true_m_rel,3)))
        #plt.legend()
        plt.xlabel("Wavelength ($\mu$m)", fontsize=18)
        plt.ylabel("Normalized Filter Transmission\nor Surface Flux (erg/cm$^{2}$/s/$\AA$)", fontsize=18)
        plt.xticks(np.arange(0, 5, step=0.5), rotation=45, fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig("junk_filter_"+filter+"_global.pdf")
        plt.close()

    return M_star
