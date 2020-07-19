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
    plot_file_name = "fyi_completion.pdf"
    plt.savefig(plot_file_name)
    print("Saved figure " + plot_file_name)

    # make a new DataFrame from a subset of the data
    # contour_data = df[["dist_asec","comp_ampl","D_xsec_strip_w_planets_rel_to_strip_1"]]
    contour_data = df

    # add column of delta_mags (necessary?)
    #contour_data["del_mag"] = 2.5*np.log10(contour_data["comp_ampl"])

    # number of comparisons: these are using 5 stripes, whose E and W halves we
    # will each use for comparison, minus the case where a half-stripe is compared
    # with itself, for a total of 9 comparions
    num_stripes = 5
    num_comparisons = 2*num_stripes-1

    # initialize cube to hold the non-interpolated KS statistic
    # (one slice for each (amplitude,radius) grid; the number of slices
    # equals the number of comparisons)
    cube_stat_no_interpolation = np.zeros((
                        num_comparisons,
                        len(contour_data["comp_ampl"].unique()),
                        len(contour_data["dist_asec"].unique())
                        ))

    # initialize ticker for adding data to cube
    ticker_num_no_interp = int(0)

    # loop over all stripes for comparison
    # (note this includes E and W halves of each stripe)
    for i in range(0,num_stripes):

        print("num_stripe:")
        print(i)
        print("ticker_num_no_interp:")
        print(ticker_num_no_interp)

        # which stripes are we comparing with? (note that we will remove
        # the half of the one strip where the planets were injected, since
        # that would represent a comparison with itself)
        if np.logical_and(i==0,stripe_w_planet!=0):
            # compare with stripe 0, if stripe 0 is not the one with the planet,
            # and set the strings which will be needed for that comparison
            comparison_string_E = 'D_xsec_strip_w_planets_rel_to_strip_0_E'
            comparison_string_W = 'D_xsec_strip_w_planets_rel_to_strip_0_W'
        elif np.logical_and(i==1,stripe_w_planet!=1):
            # compare with stripe 1, etc.
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
        else:
            # this stripe in this iteration of the for-loop contains the planet
            if half_w_planet == "E":
                comparison_string_E = '' # zero-length string
                comparison_string_W = 'D_xsec_strip_w_planets_rel_to_other_half_same_strip_with_planet'
            elif half_w_planet == "W":
                comparison_string_E = 'D_xsec_strip_w_planets_rel_to_other_half_same_strip_with_planet'
                comparison_string_W = ''

        X_unique = np.sort(contour_data.dist_asec.unique())
        Y_unique = np.sort(contour_data.comp_ampl.unique())
        X, Y = np.meshgrid(X_unique, Y_unique)

        if (len(comparison_string_E) > 0):

            # if the eastern half of the stripe is legitimate to compare to,
            # rearrange 1-D KS statistics of that eastern half into a matrix
            Z_E = contour_data.pivot_table(index='dist_asec',
                                     columns='comp_ampl',
                                     values=comparison_string_E).T.values
            # add this slice to non-interpolated cube
            cube_stat_no_interpolation[ticker_num_no_interp,:,:] = Z_E
            ticker_num_no_interp += 1 # advance ticker
        if (len(comparison_string_W) > 0):

            # if the western half is legitimate to compare to, etc...
            Z_W = contour_data.pivot_table(index='dist_asec',
                                     columns='comp_ampl',
                                     values=comparison_string_W).T.values
            # add this slice to non-interpolated cube
            cube_stat_no_interpolation[ticker_num_no_interp,:,:] = Z_W
            ticker_num_no_interp += 1 # advance ticker

        ###################################
        ## BEGIN PLOTS

        # FYI contour plot of KS statistic, no interpolation, both E and W halves
        plt.clf()
        fig, axs = plt.subplots(1, 2)
        # underplot scatter plot of sampled points
        sp0 = axs[0].scatter(contour_data["dist_asec"],contour_data["comp_ampl"], s=1)
        sp1 = axs[1].scatter(contour_data["dist_asec"],contour_data["comp_ampl"], s=1)
        # plot contour plots

        if (len(comparison_string_E) > 0):
            cp1_E = axs[0].contour(X, Y, Z_E)
            # overplot the critical line (which is always the same, regardless of strip being compared)
            df_levels = df.drop_duplicates(subset="val_xsec_crit_strip_w_planets_rel_to_strip_1_E",
                                           keep="first",
                                           inplace=False)
            levels = df_levels["val_xsec_crit_strip_w_planets_rel_to_strip_1_E"].values
            cp2_E = axs[0].contour(X, Y, Z_E, levels = levels)
            axs[0].clabel(cp2_E, inline=True, fontsize=10)
            title_E = axs[0].set_title("E")
        if (len(comparison_string_W) > 0):
            cp1_W = axs[1].contour(X, Y, Z_W)
            # overplot the critical line (which is always the same, regardless of strip being compared)
            df_levels = df.drop_duplicates(subset="val_xsec_crit_strip_w_planets_rel_to_strip_1_W",
                                           keep="first",
                                           inplace=False)
            levels = df_levels["val_xsec_crit_strip_w_planets_rel_to_strip_1_W"].values
            cp2_W = axs[1].contour(X, Y, Z_W, levels = levels)
            axs[1].clabel(cp2_W, inline=True, fontsize=10)
            title_W = axs[1].set_title("W")
        axs[0].set_xlabel("dist_asec")
        axs[1].set_xlabel("dist_asec")
        axs[0].set_ylabel("companion_ampl")
        plot_file_name = "fyi_comp_w_contours_comparison_stripe_w_planet_"+str(int(stripe_w_planet))+\
            "_half_w_planet_"+str(half_w_planet)+\
            "_comparison_stripe_"+str(int(i))+"_of_"+str(int(num_stripes-1))+".pdf"
        plt.suptitle("Comparison with stripe "+str(int(i)))
        plt.savefig(plot_file_name)
        print("Saved " + str(plot_file_name))
        plt.close()

        ## END PLOTS INSIDE FOR-LOOP
        ###################################

    # take an average across the cube
    cube_stat_no_interp_avg = np.mean(cube_stat_no_interpolation, axis=0)

    # map contrast curve to magnitudes
    Y_mag = -2.5*np.log10(Y)
    comp_ampl_mag = -2.5*np.log10(df["comp_ampl"])

    # pickle this data (I want to separately read these in and take an average
    # over averages)
    import pickle
    data_dict = { "x_axis": X,
                    "Y_axis": Y_mag,
                    "cube_stat_no_interp_avg": cube_stat_no_interp_avg }
    id_string = read_csv_basename.split(".")[0] # to distinguish this from others
    pickle_file_name = "ks_data_"+id_string+".p"
    pickle.dump( data_dict, open( pickle_file_name, "wb" ) )
    print("Saved pickle file " + pickle_file_name)
    # and some other vital stats
    vital_stats_dict = { "comp_ampl_mag": comp_ampl_mag, "df_stuff": df}
    pickle_file_name2 = "vital_stats_"+id_string+".p"
    pickle.dump( vital_stats_dict, open( pickle_file_name2, "wb" ) )
    print("Saved pickle file " + pickle_file_name2)

    '''
    # generate 2D contour plots for each individual slice, and then for the average
    levels = df_levels["val_xsec_crit_strip_w_planets_rel_to_strip_1_E"].values
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
    '''
    # now plot the average
    plt.clf()
    #cp3 = plt.contour(X, Y_mag, cube_stat_no_interp_avg, alpha = 0.5)
    #cp3.levels
    cp3 = plt.contour(X, Y_mag, cube_stat_no_interp_avg, alpha = 0.5)
    plt.clabel(cp3, inline=1, fontsize=10)
    cp4 = plt.contour(X, Y_mag, cube_stat_no_interp_avg, levels = levels, linewidths=5, color="k")
    plt.clabel(cp4, inline=1, fontsize=10)
    plt.scatter(df["dist_asec"],comp_ampl_mag, s=1)
    plt.gca().invert_yaxis()
    plt.xlabel("R (arcsec)", fontsize=18)
    plt.ylabel("$\Delta$m", fontsize=18)
    plt.xlim([0,0.55])
    plt.ylim([6,2])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    filename4 = "contour_avg_stripe_w_planet_"+str(stripe_w_planet)+"_half_w_planet_"+str(half_w_planet)+".pdf"
    #plt.show()
    plt.savefig(filename4)
    print("Wrote " + filename4)

    # extract the contour information
    p_info = cp4.collections[0].get_paths()[0]
    v = p_info.vertices
    x = v[:,0] # radius (asec)
    y = v[:,1] # delta_m (mag)
    dict_pre_df = {"x": x, "y": y}
    contour_info_df = pd.DataFrame(data=dict_pre_df)
    csv_name = "lambda_B_cc_stripe_w_planet_"+str(stripe_w_planet)+"_half_w_planet_"+str(half_w_planet)+".csv"
    contour_info_df.to_csv(csv_name)
    print("Wrote " + csv_name)

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
