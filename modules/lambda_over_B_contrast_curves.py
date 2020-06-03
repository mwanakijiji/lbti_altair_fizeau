# Generate contrast curves in the lambda/B regime from the KS analysis done on the
# ADI frames

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def main(stripe_w_planet,half_w_planet,read_csv_basename):
    '''
    INPUTS:
    stripe_w_planet: integer which sets the strip with planets injected along the median angle
        (choices are [0,1,2,3,4])
    half_w_planet: the East/West half of the stripe with the planet where the
        planet actually lies (choices are [E/W])
    read_csv_basename: file name of csv which contains the KS test data
    '''

    df = pd.read_csv(read_csv_basename)

    # print out a FYI plot, so that the user can see level of completion
    # represented by the files available
    plt.clf()
    plt.scatter(df["dist_asec"],df["comp_ampl"])
    plt.xlabel("dist (asec)")
    plt.ylabel("comp ampl")
    plt.savefig("fyi_completion.pdf")

    # make a new DataFrame from a subset of the data
    # contour_data = df[["dist_asec","comp_ampl","D_xsec_strip_w_planets_rel_to_strip_1"]]
    contour_data = df

    # add column of delta_mags (necessary?)
    #contour_data["del_mag"] = 2.5*np.log10(contour_data["comp_ampl"])

    # number of comparisons: these are using 5 stripes, whose E and W halves we
    # will each use for comparison for a total of 10 comparions (but only 9
    # really matter, because the planet is in one of them)
    num_stripes = 5
    num_comparisons = 2*num_stripes

    # initialize cube to hold the non-interpolated KS statistic
    cube_stat_no_interpolation = np.zeros((
                        num_comparisons,
                        len(contour_data["comp_ampl"].unique()),
                        len(contour_data["dist_asec"].unique())
                        ))

    # initialize cube to hold *interpolated* KS statistic
    ngridx = 100
    ngridy = 200
    cube_stat_interp = np.zeros((
                        num_comparisons,
                        ngridy,
                        ngridx
                        ))
                        
    import ipdb; ipdb.set_trace()

    # initialize ticker to adding data to cube
    ticker_num = int(0)

    # loop over all stripes
    for i in range(0,num_stripes):

        # which stripes are we comparing with? (note that we will remove
        # the half of the one strip where the planets were injected, since
        # that would represent a comparison with itself)
        if np.logical_and(i==0,stripe_w_planet!=0):
            #if (half_w_planet == "E"):
            #if (half_w_planet == "W"):
            comparison_string_E = 'D_xsec_strip_w_planets_rel_to_strip_0_E'
            comparison_string_W = 'D_xsec_strip_w_planets_rel_to_strip_0_W'
        elif np.logical_and(i==1,stripe_w_planet!=1):
            comparison_string_E = 'D_xsec_strip_w_planets_rel_to_strip_1_E'
            comparison_string_W = 'D_xsec_strip_w_planets_rel_to_strip_1_W'
        elif np.logical_and(i==2,stripe_w_planet!=2):
            comparison_string_E = 'D_xsec_strip_w_planets_rel_to_strip_2_E'
            comparison_string_W = 'D_xsec_strip_w_planets_rel_to_strip_2_W'
        elif np.logical_and(i==3,stripe_w_planet!=3):
            comparison_string_E = 'D_xsec_strip_w_planets_rel_to_strip_3_E'
            comparison_string_W = 'D_xsec_strip_w_planets_rel_to_strip_3_W'
        elif np.logical_and(i==4,stripe_w_planet!=4):
            comparison_string_E = 'D_xsec_strip_w_planets_rel_to_strip_4_E'
            comparison_string_W = 'D_xsec_strip_w_planets_rel_to_strip_4_W'
        else: # this stripe in this iteration of the for-loop contains the planet
            if half_w_planet == "E":
                comparison_string_E = np.nan
                comparison_string_W = 'D_xsec_strip_w_planets_rel_to_other_half_same_strip_with_planet'
            elif half_w_planet == "W":
                comparison_string_E = 'D_xsec_strip_w_planets_rel_to_other_half_same_strip_with_planet'
                comparison_string_W = np.nan

        if (comparison_string_E != np.nan):
            # rearrange 1-D KS statistics into a matrix
            Z_E = contour_data.pivot_table(index='dist_asec',
                                     columns='comp_ampl',
                                     values=comparison_string_E).T.values
        elif (comparison_string_W != np.nan):
            Z_W = contour_data.pivot_table(index='dist_asec',
                                     columns='comp_ampl',
                                     values=comparison_string_W).T.values

        X_unique = np.sort(contour_data.dist_asec.unique())
        Y_unique = np.sort(contour_data.comp_ampl.unique())
        X, Y = np.meshgrid(X_unique, Y_unique)

        # add this slice to non-interpolated cube
        cube_stat_no_interpolation[ticker_num,:,:] = Z_E
        ticker_num += 1 # advance ticker
        cube_stat_no_interpolation[ticker_num,:,:] = Z_W
        ticker_num += 1 # advance ticker



        # linearly interpolate the data we have onto a regular grid in magnitude space
        xi = np.linspace(0, 0.55, num=ngridx)
        yi = np.linspace(0, 0.3, num=ngridy)

        # Linearly interpolate the data (X, Y) on a grid defined by (xi, yi).
        triang_2 = tri.Triangulation(contour_data["dist_asec"].values,
                                   contour_data["comp_ampl"].values)
        interpolator = tri.LinearTriInterpolator(triang_2, contour_data[comparison_string].values)
        Xi, Yi = np.meshgrid(xi, yi)
        zi = interpolator(Xi, Yi)

        # add this slice to interpolated cube
        cube_stat_interp[i,:,:] = zi

        ###################################
        ## BEGIN PLOTS

        # FYI contour plot of KS statistic, no interpolation
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # underplot scatter plot of sampled points
        sp = ax.scatter(contour_data["dist_asec"],contour_data["comp_ampl"], s=1)
        # plot a contour plot
        cp1 = ax.contour(X, Y, Z)
        # overplot the critical line
        df_levels = df.drop_duplicates(subset="val_xsec_crit_strip_w_planets_rel_to_strip_1",
                                       keep="first",
                                       inplace=False)
        levels = df_levels["val_xsec_crit_strip_w_planets_rel_to_strip_1"].values
        cp2 = ax.contour(X, Y, Z, levels = levels)
        ax.set_xlabel("dist_asec")
        ax.set_ylabel("companion_ampl")
        plt.savefig("fyi_comp_w_contours_comparison_"+str(int(i))+"_of_"+str(int(num_stripes))+".pdf")

        # FYI contour plot of KS statistic, WITH interpolation
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # underplot scatter plot of sampled points
        sp = ax.scatter(contour_data["dist_asec"],contour_data["comp_ampl"], s=1)
        # plot a contour plot
        cp1 = ax.contour(Xi, Yi, zi)
        # overplot the critical line
        df_levels = df.drop_duplicates(subset="val_xsec_crit_strip_w_planets_rel_to_strip_1",
                                       keep='first',
                                       inplace=False)
        levels = df_levels["val_xsec_crit_strip_w_planets_rel_to_strip_1"].values
        cp2 = ax.contour(Xi, Yi, zi, levels = levels)

        ax.set_xlabel("dist_asec")
        ax.set_ylabel("companion_ampl")

        # save FYI plot
        filename2 = "junk_" + str(int(i)) + "_with_interpolation.png"
        plt.savefig(filename2)
        print("Saved " + filename2)

        print("------------")

        ## END PLOTS INSIDE FOR-LOOP
        ###################################

    # take an average across the cube

    cube_stat_avg = np.mean(cube_stat, axis=0)

    # map contrast curve to magnitudes
    Y_mag = -2.5*np.log10(Y)
    comp_ampl_mag = -2.5*np.log10(df["comp_ampl"])

    # generate 2D contour plots for each individual slice, and then for the average
    levels = df_levels["val_xsec_crit_strip_w_planets_rel_to_strip_1"].values
    for t in range(0,4):
        # loop over stripe comparisons
        plt.clf()
        cp3 = plt.contour(X, Y_mag, cube_stat[t,:,:], alpha = 0.5)
        cp4 = plt.contour(X, Y_mag, cube_stat[t,:,:], levels = levels, linewidths=5, color="k")
        plt.scatter(df["dist_asec"],comp_ampl_mag, s=1)
        plt.gca().invert_yaxis()
        plt.xlabel("R (arcsec)")
        plt.ylabel("$\Delta$m")
        plt.xlim([0,0.55])
        plt.ylim([5.2,1])

        filename3 = "contour_mags_" + str(int(t)) + ".png"
        plt.savefig(filename3)
        print("Saved " + filename3)

    # now plot the average
    plt.clf()
    cp3 = plt.contour(X, Y_mag, cube_stat_avg, alpha = 0.5)
    cp4 = plt.contour(X, Y_mag, cube_stat_avg, levels = levels, linewidths=5, color="k")
    plt.scatter(df["dist_asec"],comp_ampl_mag, s=1)
    plt.gca().invert_yaxis()
    plt.xlabel("R (arcsec)")
    plt.ylabel("$\Delta$m")
    plt.xlim([0,0.55])
    plt.ylim([5.2,1])
    filename4 = "contour_avg.png"
    plt.savefig(filename4)
    print(filename4)

    # FYI 2D color plot
    '''
    plt.imshow(cube_stat_avg, origin="lower")
    plt.xlabel("radius (arbit units)")
    plt.ylabel("comp amplitude (arbit units)")
    plt.show()
    '''

    # FYI linear contrast curve
    '''
    levels = df_levels["val_xsec_crit_strip_w_planets_rel_to_strip_1"].values
    cp23 = plt.contour(X, Y, cube_stat_avg, levels = levels)
    plt.scatter(df["dist_asec"],df["comp_ampl"], s=1)
    '''

    # make FYI scatter plots of KS statistic as function of radius, and overplot each
    # strip comparison for a given companion amplitude
    '''
    for i in range(0,np.shape(cube_stat)[1]):
        # loop over companion amplitudes
        plt.clf()
        for j in range(0,4):
            # loop over different stripe comparisons
            plt.plot(contour_data["dist_asec"].drop_duplicates(),cube_stat[j,i,:],marker="o")
        plt.axhline(y=df["val_xsec_crit_strip_w_planets_rel_to_strip_1"][0])
        plt.show()
    '''
