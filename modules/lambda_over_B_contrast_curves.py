# Generate contrast curves in the lambda/B regime from the KS analysis done on the
# ADI frames

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os
import glob

def main(read_csvs_directory, write_stem):
    '''
    INPUTS:
    read_csvs_directory: directory in which the csv files to average reside (these
        csvs should correspond to one cardinal direction (for example, files
        corresponding to 0W, 1W, 2W, 3W, and 4W; but not non-W directions)
    write_stem: string to append to written data (like PNGs)
    '''

    # get the names of all the csvs
    read_csv_file_names = list(glob.glob(read_csvs_directory+"/*.csv"))

    # read in test file to initialize cube
    df_test = pd.read_csv(read_csv_file_names[0])
    # initialize meshgrid of (amplitude, radius) space
    X_unique = np.sort(df_test.dist_asec.unique())
    Y_unique = np.sort(df_test.comp_ampl.unique())
    X, Y = np.meshgrid(X_unique, Y_unique)

    # initialize master cube needed for making an average of averages
    # (each slice corresponds to all the comparisons of all other half-strips
    # with a given half-strip; i.e., each slice corresponds to one CSV)
    master_KS_cube = np.zeros((
                        len(read_csv_file_names),
                        len(df_test["comp_ampl"].unique()),
                        len(df_test["dist_asec"].unique())
                        ))

    # loop over all CSVs
    for file_num in range(0,len(read_csv_file_names)):

        df = pd.read_csv(read_csv_file_names[file_num])

        # print out a FYI plot, so that the user can see level of completion
        # represented by the files available
        plt.clf()
        plt.scatter(df["dist_asec"],df["comp_ampl"])
        plt.xlabel("dist (asec)")
        plt.ylabel("comp ampl")
        plot_file_name = "fyi_completion"+os.path.basename(read_csv_file_names[file_num]).split(".")[0]+".pdf"
        plt.title("Completion of\n"+str(os.path.basename(read_csv_file_names[file_num])))
        plt.savefig(plot_file_name)
        print("Saved figure " + plot_file_name)

        # make a new DataFrame from a subset of the data
        # contour_data = df[["dist_asec","comp_ampl","D_xsec_strip_w_planets_rel_to_strip_1"]]
        contour_data = df

        # add column of delta_mags (necessary?)
        #contour_data["del_mag"] = 2.5*np.log10(contour_data["comp_ampl"])

        # number of comparisons for a given half-stripe
        num_stripes = 5
        num_comparisons = num_stripes-1

        # initialize cube to hold the non-interpolated KS statistic
        # (one slice for each (amplitude,radius) grid; the number of slices
        # equals the number of comparisons; note that if some comparisons are
        # neglected--like comparisons with neighboring stripes--then the
        # cube will not become entirely filled with data; but the nanmean
        # at the end effectively ignores the slices which are unchanged from
        # NaNs)
        cube_stat_no_interpolation = np.nan*np.ones((
                            num_comparisons,
                            len(contour_data["comp_ampl"].unique()),
                            len(contour_data["dist_asec"].unique())
                            ))

        # initialize ticker for adding data to cube
        ticker_num_no_interp = int(0)

        # we are comparing one half-strip to the other four in that bunch,
        # pointed in the same cardinal direction (i.e., 2E is compared with
        # 0E, 1E, 3E, 4E)
        if np.logical_or(
            np.logical_or(("0E" in read_csv_file_names[file_num]),("0W" in read_csv_file_names[file_num])),
            np.logical_or(("0VN" in read_csv_file_names[file_num]),("0VS" in read_csv_file_names[file_num]))
            ):
            # set the strings for comparing the other strips with 0
            this_stripe_string = 'strip_0'
            crit_contour_string = 'val_crit_strip_w_planets_rel_to_strip_0'
            array_comparison_strings = ['D_strip_w_planets_rel_to_strip_2',
                                        'D_strip_w_planets_rel_to_strip_3',
                                        'D_strip_w_planets_rel_to_strip_4']
        elif np.logical_or(
            np.logical_or(("1E" in read_csv_file_names[file_num]),("1W" in read_csv_file_names[file_num])),
            np.logical_or(("1VN" in read_csv_file_names[file_num]),("1VS" in read_csv_file_names[file_num]))
            ):
            # set the strings for comparing the other strips with 0
            this_stripe_string = 'strip_1'
            crit_contour_string = 'val_crit_strip_w_planets_rel_to_strip_1'
            array_comparison_strings = ['D_strip_w_planets_rel_to_strip_3',
                                        'D_strip_w_planets_rel_to_strip_4']
        elif np.logical_or(
            np.logical_or(("2E" in read_csv_file_names[file_num]),("2W" in read_csv_file_names[file_num])),
            np.logical_or(("2VN" in read_csv_file_names[file_num]),("2VS" in read_csv_file_names[file_num]))
            ):
            # set the strings for comparing the other strips with 0
            this_stripe_string = 'strip_2'
            crit_contour_string = 'val_crit_strip_w_planets_rel_to_strip_2'
            array_comparison_strings = ['D_strip_w_planets_rel_to_strip_0',
                                        'D_strip_w_planets_rel_to_strip_4']
        elif np.logical_or(
            np.logical_or(("3E" in read_csv_file_names[file_num]),("3W" in read_csv_file_names[file_num])),
            np.logical_or(("3VN" in read_csv_file_names[file_num]),("3VS" in read_csv_file_names[file_num]))
            ):
            # set the strings for comparing the other strips with 0
            this_stripe_string = 'strip_3'
            crit_contour_string = 'val_crit_strip_w_planets_rel_to_strip_3'
            array_comparison_strings = ['D_strip_w_planets_rel_to_strip_0',
                                        'D_strip_w_planets_rel_to_strip_1']
        elif np.logical_or(
            np.logical_or(("4E" in read_csv_file_names[file_num]),("4W" in read_csv_file_names[file_num])),
            np.logical_or(("4VN" in read_csv_file_names[file_num]),("4VS" in read_csv_file_names[file_num]))
            ):
            # set the strings for comparing the other strips with 0
            this_stripe_string = 'strip_4'
            crit_contour_string = 'val_crit_strip_w_planets_rel_to_strip_4'
            array_comparison_strings = ['D_strip_w_planets_rel_to_strip_0',
                                        'D_strip_w_planets_rel_to_strip_1',
                                        'D_strip_w_planets_rel_to_strip_2']
        print("This stripe string: \n"+this_stripe_string)
        # put the KS grids into the cube whose slices correspond to
        # comparisons with this particular half-stripe
        #import ipdb; ipdb.set_trace()
        for comparison_num in range(0,len(array_comparison_strings)):

            # initialize meshgrid of (amplitude, radius) space
            X_unique = np.sort(contour_data.dist_asec.unique())
            Y_unique = np.sort(contour_data.comp_ampl.unique())
            X, Y = np.meshgrid(X_unique, Y_unique)

            # rearrange 1-D KS statistics into a matrix
            Z_E = contour_data.pivot_table(index='dist_asec',
                                     columns='comp_ampl',
                                     values=array_comparison_strings[comparison_num]).T.values
            # add this slice to non-interpolated cube
            cube_stat_no_interpolation[comparison_num,:,:] = Z_E

            ###################################
            ## BEGIN PLOTS

            # FYI contour plot of KS statistic, no interpolation, both E and W halves
            plt.clf()
            fig, axs = plt.subplots(1, 1)
            # underplot scatter plot of sampled points
            sp0 = axs.scatter(contour_data["dist_asec"],contour_data["comp_ampl"], s=1)

            # plot contour plots
            cp1_E = axs.contour(X, Y, Z_E)
            # overplot the critical line (which is always the same, regardless of strip being compared)
            levels = [df[crit_contour_string].iloc[0]]
            cp2_E = axs.contour(X, Y, Z_E, levels = levels, linewidths=8, alpha = 0.5)
            axs.clabel(cp2_E, inline=True, fontsize=10)

            axs.set_xlabel("dist_asec")
            axs.set_ylabel("companion_ampl")
            plot_file_name = "fyi_KS_contours_"+\
                            this_stripe_string + "_" +\
                            "using_comparison_"+\
                            str(array_comparison_strings[comparison_num])+\
                            "_based_on_data_"+write_stem+".pdf"
            plt.suptitle(this_stripe_string +\
                        "\ncompared with "+\
                        array_comparison_strings[comparison_num])
            plt.savefig(plot_file_name)
            print("Saved " + str(plot_file_name))
            plt.close()
            #import ipdb; ipdb.set_trace()

            ## END PLOTS INSIDE FOR-LOOP
            ###################################

        # take an average across the cube
        cube_stat_no_interp_avg = np.nanmean(cube_stat_no_interpolation, axis=0)

        # add that average as a slice to the master cube
        master_KS_cube[file_num,:,:] = cube_stat_no_interp_avg

        # map contrast curve to magnitudes
        Y_mag = -2.5*np.log10(Y) # the meshgrid
        comp_ampl_mag = -2.5*np.log10(df["comp_ampl"]) # the companion amplitudes

        # now plot the average of the comparisons to this particular half-stripe
        plt.clf()
        #cp3 = plt.contour(X, Y_mag, cube_stat_no_interp_avg, alpha = 0.5)
        #cp3.levels
        cp3 = plt.contour(X, Y_mag, cube_stat_no_interp_avg, alpha = 0.5)
        plt.clabel(cp3, inline=1, fontsize=10)
        cp4 = plt.contour(X, Y_mag, cube_stat_no_interp_avg, levels = levels, linewidths=8, alpha = 0.5)
        #plt.clabel(cp4, inline=1, fontsize=10)
        plt.scatter(df["dist_asec"],comp_ampl_mag, s=1)
        plt.gca().invert_yaxis()
        plt.xlabel("R (arcsec)", fontsize=18)
        plt.ylabel("$\Delta$m", fontsize=18)
        plt.xlim([0,0.55])
        plt.ylim([10,0])
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        filename4 = "contour_avg_for_"+str(this_stripe_string)+".pdf"
        #plt.show()
        plt.savefig(filename4)
        print("Wrote average of comparisons to this particular half-stripe " + filename4)
        print("-----------------------------------------------------------------")


    #import ipdb; ipdb.set_trace()
    # take an average across the cube of cubes
    average_KS_2d_master = np.nanmean(master_KS_cube, axis = 0)

    # write out
    plt.clf()
    cp3 = plt.contour(X, Y_mag, average_KS_2d_master, alpha = 0.5)
    plt.clabel(cp3, inline=1, fontsize=10)
    cp4 = plt.contour(X, Y_mag, average_KS_2d_master, levels = levels, linewidths=5, color="k")
    #plt.clabel(cp4, inline=1, fontsize=10)
    plt.scatter(df["dist_asec"],comp_ampl_mag, s=1)
    plt.gca().invert_yaxis()
    plt.xlabel("R (arcsec)", fontsize=18)
    plt.ylabel("$\Delta$m", fontsize=18)
    plt.xlim([0,0.55])
    plt.ylim([6,2])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title("")
    plt.tight_layout()
    filename4 = "avg_of_avgs_" + write_stem + ".pdf"
    #plt.show()
    plt.savefig(filename4)
    print("Wrote average of averages as " + filename4)


    # extract the contour information
    p_info = cp4.collections[0].get_paths()[0]
    v = p_info.vertices
    x = v[:,0] # radius (asec)
    y = v[:,1] # delta_m (mag)
    dict_pre_df = {"x": x, "y": y}
    contour_info_df = pd.DataFrame(data=dict_pre_df)
    csv_name = "lambda_B_cc_stripe_w_planet_"+str(write_stem)+".csv"
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
