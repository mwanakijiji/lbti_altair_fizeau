import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from modules import *
from matplotlib.transforms import Transform
from matplotlib.ticker import (
    AutoLocator, AutoMinorLocator, FuncFormatter)
from scipy import interpolate

## ## THESE SHOULD BE IN INIT
# define some basic functions
def asec_to_AU(input_asec):

    # convert asec to AU
    dist_altair = 5.130 # pc
    output_AU = np.multiply(dist_altair,input_asec)

    return output_AU


def AU_to_asec(input_AU):

    # convert AU to asec
    dist_altair = 5.130 # pc
    output_asec = np.divide(input_AU,dist_altair)

    return output_asec

# absolute magnitude of host star in NB205 filter

## ## PLACEHOLDER VALUE HERE; CURRENT VALUE FROM NOTEBOOK
## ## determine_abs_mag_altair.ipynb; NOT DOUBLE-CHECKED;
## ## REPLACE LATER (NOTE ALSO THAT ALTAIR IS A VARIABLE SOURCE!)
def calculate_abs_mag():

    abs_mag_altair_nb405 = 1.87

    return abs_mag_altair_nb405


def linear_2_mass(df_pass, star_abs_mag_pass):
    '''
    Takes linear contrast and
    1. converts it to magnitude difference
    2. finds abs magnitude equivalent to the sensitivity threshold
    3. converts abs magnitude to masses for a given model

    INPUTS:
    df_pass: dataframe containing "asec" and "contrast_lin" (after any small
        angle correction, if applicable)
    star_abs_mag_pass: absolute magnitude of the star through the bandpass

    RETURNS:
    df_w_masses: dataframe same as the input, but with a column for masses
    '''

    # make a copy of the input dataframe
    df_new = df_pass.copy(deep=True)

    # convert linear empirical contrast to del_mag
    df_new["del_mag_LMIR"] = -2.5*np.log10(df_new["contrast_lin"])

    # convert del_mag (between planet and host star) to abs. mag (of planet)
    df_new["abs_mag_LMIR"] = np.add(df_new["del_mag_LMIR"],star_abs_mag_pass)

    ## ## SHOULD BE IN __INIT__
    # convert asec to AU
    dist_altair = 5.130 # pc
    df_new["AU"] = np.multiply(dist_altair,df_new["asec"])

    # read in models

    # AMES-Cond: https://phoenix.ens-lyon.fr/Grids/AMES-Cond/ISOCHRONES/model.AMES-Cond-2000.M-0.0.NaCo.Vega
    # Br-alpha filter is model_data["NB4.05"], in Vega magnitudes

    model_data = pd.read_csv("./notebooks_for_development/data/1gr_data.txt",
        delim_whitespace=True)
    #print(model_data)
    # read in NACO transmission curve for comparison
    naco_trans = pd.read_csv("./notebooks_for_development/data/Paranal_NACO.NB405.dat.txt",
        names = ["angstrom", "transm"], delim_whitespace=True)
    lmir_bralpha_trans = pd.read_csv("./notebooks_for_development/data/br-alpha_NDC.txt",
        delim_whitespace=True)
    lmir_bralpha_trans["Wavelength_angstr"] = np.multiply(10000.,lmir_bralpha_trans["Wavelength"])

    # plot filter curves
    '''
    plt.clf()
    plt.plot(naco_trans["angstrom"], naco_trans["transm"], label = "NACO NB4.05")
    plt.plot(lmir_bralpha_trans["Wavelength_angstr"], lmir_bralpha_trans["Trans_77"],
         label = "LMIR Br-"+r"$\alpha$ (T = 77 K)")
    plt.xlim([39750,41550])
    plt.xlabel("Wavelength ("+r"$\AA$"+")")
    plt.ylabel("Transmission")
    plt.legend()
    plt.savefig("junk.pdf")
    '''

    # ### Interpolate the models to map absolute mag to mass
    # make function to interpolate models
    f_abs_mag_2_mass = interpolate.interp1d(model_data["NB4.05"],model_data["M/Ms"],kind="linear")
    # ... and its inverse
    f_mass_2_abs_mag = interpolate.interp1d(model_data["M/Ms"],model_data["NB4.05"],kind="linear")

    # return masses (M/M_solar) corresponding to our contrast curve
    df_new["masses_LMIR"] = f_abs_mag_2_mass(df_new["del_mag_widthFWHM"])

    print(df_new["masses_LMIR"])

    # plot model data and interpolation
    plt.clf()
    plt.plot(model_data["NB4.05"], model_data["M/Ms"], color="blue", label="model points", marker="o")
    plt.scatter(df_new["abs_mag_LMIR"], df_new["masses_LMIR"], color="orange",
            label="contrast curve interpolation")
    plt.xlim([0,20])
    plt.xlabel("abs_mag LMIR")
    plt.ylabel("M/M_solar")
    plt.legend()
    plt.savefig("junk.pdf")

    # return more masses corresponding to interpolations at intervals
    mass_intervals = [0.5,0.6,0.7,0.8,0.9,1.0]
    annotate_mass_intervals = ["0.5 Ms","0.6 Ms","0.7 Ms","0.8 Ms","0.9 Ms","1.0 Ms"]
    abs_mag_intervals = f_mass_2_abs_mag(mass_intervals)

    # ### Make plot
    # #### left y-axis: abs mag
    # #### bottom x-axis: asec
    # #### right y-axis: M/Ms
    # #### top x-axis: AU

    #f = lambda q: q
    #finv = lambda x: np.log10(2+x)+np.cos(x)
    '''
    fig, ax = plt.subplots()
    fig.suptitle("Contrast curve\n(based on M_altair = 1.8; NOT QUADRUPLE-CHECKED")
    ax2 = ax.twinx()
    ax.set_xlim([0,2.2]) # 0 to 2.2 asec
    ax.get_shared_y_axes().join(ax,ax2)

    ax.set_ylabel('Abs mag (LMIR)')
    ax.set_xlabel('Angle (asec)')

    ax.plot(contrast_df["asec"], contrast_df["abs_mag_LMIR"])

    # secondary x axis on top
    secax_x = ax.secondary_xaxis('top', functions=(asec_to_AU, AU_to_asec))
    secax_x.set_xlabel('Distance (AU)')

    # draw horizontal lines corresponding to certain masses
    for t in range(0,len(mass_intervals)):
        ax.axhline(y=abs_mag_intervals[t], linestyle="--", color="k")
        ax.annotate(annotate_mass_intervals[t],
                xy=(0.4,abs_mag_intervals[t]),
                xytext=(0,0), textcoords="offset points")

        ax2.yaxis.set_major_formatter(FuncFormatter(lambda t,pos: f"{f_abs_mag_2_mass(t):.2f}"))
        ax2.set_ylabel('Masses (M/Ms)')
        plt.gca().invert_yaxis()
        plt.show()
    '''


def main():
    '''
    Take an input 1D contrast curve and convert it to masses
    '''

    # configuration data
    config = configparser.ConfigParser() # for parsing values in .init file
    config.read("/modules/config.ini")

    # get abs magnitude of Altair
    star_abs_mag = calculate_abs_mag()

    # make/read in a contrast curve, where contrast is defined as the flux ratio
    # F_planet/F_star where detection has 5-sigma significance
    # names: "contrast_lin" and "asec"
    contrast_df = pd.read_csv("./notebooks_for_development/data/placeholder_classical_curve_20200316.csv")

    df_w_masses = linear_2_mass(df_pass = contrast_df, star_abs_mag_pass = star_abs_mag)
    print(star_abs_mag)

    # sources of error:
    # 1. uncertainty of distance from parallax
    # 2. wavelength dependency of atmospheric transmission -> absolute magnitude of planet
    # 3. small differences in filter bandpass between LMIR, NB4.05
