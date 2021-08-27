#!/usr/bin/env python
# coding: utf-8

# This calculates order-of-magnitude-accurate planet/star contrasts in
# thermal equilibrium

# Created 2021 Aug. 24 by E.S.

import numpy as np
from astropy.constants import k_B, h, c
from astropy import units as u

lambda_obs = 4.05e-6 * u.meter # wavelength of observation (m)
R_p = 1. * u.Rjup # radius of planet (R_J)
R_star = 9.96 * 1.65 * u.Rjup # radius of star (R_J); (1.65 R_sol) * (9.96 R_J/R_sol)
Teff_star = 6850 * u.K # effective temp of star
a_orbit = 2. * u.au# semimajor axis

f_prime = 0.25 # 1/4: uniform redistribution; 2/3: instant re-emission
A_B = 0.3 # Bond albedo


def planet_temp(Teff_star_pass,R_star_pass,f_prime_pass,A_B_pass,a_pass):
    # finds equilibrium temperature of planet
    # Eqn. 3.9 in Seager, split into
    # part1 * part2

    # note extra conversion to m to avoid mixed units
    part1 = Teff_star_pass*np.sqrt(np.divide(R_star_pass.to(u.m),a_pass.to(u.m)))
    part2 = np.power(f_prime_pass*(1.-A_B_pass),0.25)

    Teq_p_pass = part1*part2

    return Teq_p_pass


def contrast(wavel_pass,Teff_star_pass,Teq_p_pass,R_star_pass,R_p_pass):
    # finds planet/star contrast in thermal equilibrium
    # Eqn. 3.41 in Seager, split into
    # [part1/part2] * (part3)

    nu = np.divide(c,lambda_obs)

    part1 = np.exp(np.divide(h*nu,k_B*Teff_star))-1
    part2 = np.exp(np.divide(h*nu,k_B*Teq_p_pass))-1
    part3 = np.divide(np.power(R_p,2),np.power(R_star,2))

    contrast_val = np.divide(part1,part2)*part3

    return contrast_val


T_eq_found = planet_temp(Teff_star_pass=Teff_star,
                        R_star_pass=R_star,
                        f_prime_pass=f_prime,
                        A_B_pass=A_B,
                        a_pass=a_orbit)


contrast_found = contrast(wavel_pass=lambda_obs,
                Teff_star_pass=Teff_star,
                Teq_p_pass=T_eq_found,
                R_star_pass=R_star,
                R_p_pass=R_p)


print("Found contrast: ")
print(contrast_found)
