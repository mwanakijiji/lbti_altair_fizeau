#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This is to develop the code the generate contrast curves
# in the lambda/B regime from the KS analysis done on the 
# ADI frames

# Created 2020 May 28 by E.S.


# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri

get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


df = pd.read_csv("data/test_lambda_over_b_1.csv")


# In[6]:


# print out a plot, so that the user can see level of completion 
# represented by the files available

plt.clf()
plt.scatter(df["dist_asec"],df["comp_ampl"])
plt.show()


# In[7]:


# make a new DataFrame from a subset of the data
# contour_data = df[["dist_asec","comp_ampl","D_xsec_strip_w_planets_rel_to_strip_1"]]
contour_data = df

# initialize cube to hold the KS statistic
cube_stat = np.zeros((4,len(contour_data["comp_ampl"].unique()),len(contour_data["dist_asec"].unique())))

for i in range(1,5):
    
    # which stripes are we comparing with?
    if (i==1):
        comparison_string = 'D_xsec_strip_w_planets_rel_to_strip_1'
    elif (i==2):
        comparison_string = 'D_xsec_strip_w_planets_rel_to_strip_2'
    elif (i==3):
        comparison_string = 'D_xsec_strip_w_planets_rel_to_strip_3'
    elif (i==4):
        comparison_string = 'D_xsec_strip_w_planets_rel_to_strip_4'
    
    # arrange 1-D arrangement of KS statistics into a matrix
    Z = contour_data.pivot_table(index='dist_asec', 
                                 columns='comp_ampl', 
                                 values=comparison_string).T.values

    X_unique = np.sort(contour_data.dist_asec.unique())
    Y_unique = np.sort(contour_data.comp_ampl.unique())
    X, Y = np.meshgrid(X_unique, Y_unique)
    
    # add this slice to cube
    cube_stat[i-1,:,:] = Z

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # underplot scatter plot of sampled points
    sp = ax.scatter(contour_data["dist_asec"],contour_data["comp_ampl"], s=1)

    # plot a contour plot
    cp1 = ax.contour(X, Y, Z)

    # overplot the critical line
    df_levels = df.drop_duplicates(subset="val_xsec_crit_strip_w_planets_rel_to_strip_1", 
                                   keep='first', 
                                   inplace=False)
    levels = df_levels["val_xsec_crit_strip_w_planets_rel_to_strip_1"].values
    cp2 = ax.contour(X, Y, Z, levels = levels)

    ax.set_xlabel("dist_asec")
    ax.set_ylabel("companion_ampl")

    plt.show()
    #plt.savefig("junk_comp_w_4.pdf")


# In[40]:


## MAKE INTERPOLATION 
##

# make a new DataFrame from a subset of the data
# contour_data = df[["dist_asec","comp_ampl","D_xsec_strip_w_planets_rel_to_strip_1"]]
contour_data = df

# add column of delta_mags
contour_data["del_mag"] = 2.5*np.log10(contour_data["comp_ampl"])

# initialize cubes to hold the KS statistic
cube_stat = np.zeros((4,len(contour_data["comp_ampl"].unique()),len(contour_data["dist_asec"].unique())))

# interpolated cube
ngridx = 100
ngridy = 200
cube_stat_interp = np.zeros((4,ngridy,ngridx)) # the interpolated cube

for i in range(1,5):
    
    # which stripes are we comparing with?
    if (i==1):
        comparison_string = 'D_xsec_strip_w_planets_rel_to_strip_1'
    elif (i==2):
        comparison_string = 'D_xsec_strip_w_planets_rel_to_strip_2'
    elif (i==3):
        comparison_string = 'D_xsec_strip_w_planets_rel_to_strip_3'
    elif (i==4):
        comparison_string = 'D_xsec_strip_w_planets_rel_to_strip_4'
    
    # arrange 1-D arrangement of KS statistics into a matrix
    Z = contour_data.pivot_table(index='dist_asec', 
                                 columns='comp_ampl', 
                                 values=comparison_string).T.values

    X_unique = np.sort(contour_data.dist_asec.unique())
    Y_unique = np.sort(contour_data.comp_ampl.unique())
    X, Y = np.meshgrid(X_unique, Y_unique)
    
    # add this slice to cube
    cube_stat[i-1,:,:] = Z

    # plot
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # underplot scatter plot of sampled points
    sp = ax.scatter(contour_data["dist_asec"],contour_data["comp_ampl"], s=1)
    # plot a contour plot
    cp1 = ax.contour(X, Y, Z)
    # overplot the critical line
    df_levels = df.drop_duplicates(subset="val_xsec_crit_strip_w_planets_rel_to_strip_1", 
                                   keep='first', 
                                   inplace=False)
    levels = df_levels["val_xsec_crit_strip_w_planets_rel_to_strip_1"].values
    cp2 = ax.contour(X, Y, Z, levels = levels)

    ax.set_xlabel("dist_asec")
    ax.set_ylabel("companion_ampl")

    # save FYI plot
    filename = "junk_" + str(int(i)) + "_no_interpolation.png"
    plt.savefig(filename)
    print("Saved " + filename)
    
    ################################################
    ## now do the same, with an interpolated cube
    ## (is it necessary to do the grid in mag/log space?)
    
    # linearly interpolate the data we have onto a regular grid in magnitude space
    xi = np.linspace(0, 0.55, num=ngridx)
    yi = np.linspace(0, 0.3, num=ngridy)
    #print(contour_data[comparison_string].values)
    
    '''
    # Perform linear interpolation of the data (x,y)
    # on a grid defined by (xi,yi)
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)
    '''
    
    # Linearly interpolate the data (X, Y) on a grid defined by (xi, yi).
    triang_mag = tri.Triangulation(contour_data["dist_asec"].values, 
                               contour_data["comp_ampl"].values)
    interpolator = tri.LinearTriInterpolator(triang_mag, contour_data[comparison_string].values)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)
    print(contour_data[comparison_string])
    print(Xi)
    print(Yi)
    #plt.savefig("junk_comp_w_4.pdf")
    
    # add this slice to cube
    cube_stat_interp[i-1,:,:] = zi
    print(zi)

    # plot
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


# In[5]:


# take an average across the cube

cube_stat_avg = np.mean(cube_stat, axis=0)


# In[16]:


# FYI 2D color plot
'''
plt.imshow(cube_stat_avg, origin="lower")
plt.xlabel("radius (arbit units)")
plt.ylabel("comp amplitude (arbit units)")
plt.show()
'''


# In[7]:


# FYI linear contrast curve 

'''
levels = df_levels["val_xsec_crit_strip_w_planets_rel_to_strip_1"].values
cp23 = plt.contour(X, Y, cube_stat_avg, levels = levels)
plt.scatter(df["dist_asec"],df["comp_ampl"], s=1)
'''


# In[8]:


# map contrast curve to magnitudes

Y_mag = -2.5*np.log10(Y)
comp_ampl_mag = -2.5*np.log10(df["comp_ampl"])


# In[32]:


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
    
    filename3 = "contour_" + str(int(t)) + ".png"
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


# In[22]:


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

