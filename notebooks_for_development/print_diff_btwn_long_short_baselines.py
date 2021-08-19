import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter

# This finds difference in del_mag between short and long LBT baselines, in
# response to a reviewer's comment that the difference is not obvious on another plot

# make list of all the files
#lambda_over_D = pd.read_csv("./data/modern_contrast_curve.csv")
psf_profiles_all = pd.read_csv("./data/example_psf_profiles.csv", index_col=0)
# select subset
psf_profiles = psf_profiles_all.iloc[:, 0:10]

# lambda/D data
lambda_over_D = pd.read_csv("./data/modern_curve_20200713.csv")

# for lambda/B, there are a number of curves; we will read them all in
lambda_over_B_N = pd.read_csv("../results_20201120/lambda_B_cc_stripe_w_planet_20201021_allN.csv", skiprows=1, names=["X_N","Y_N"])
lambda_over_B_E = pd.read_csv("../results_20201120/lambda_B_cc_stripe_w_planet_20201021_allE.csv", skiprows=1, names=["X_E","Y_E"])
lambda_over_B_S = pd.read_csv("../results_20201120/lambda_B_cc_stripe_w_planet_20201021_allS.csv", skiprows=1, names=["X_S","Y_S"])
lambda_over_B_W = pd.read_csv("../results_20201120/lambda_B_cc_stripe_w_planet_20201021_allW.csv", skiprows=1, names=["X_W","Y_W"])

#print(lambda_over_B_E["y"].to_numpy())
#print(type(lambda_over_B_E["y"][0]))
#avg_del_m_long_baselines = np.mean(lambda_over_B_E["y"].to_numpy(),lambda_over_B_W["y"].to_numpy())

# make new DataFrame!!
lambda_over_B_long_baseline = pd.DataFrame.copy(lambda_over_B_E,deep=True)
lambda_over_B_long_baseline["Y_W"] = lambda_over_B_W["Y_W"]
lambda_over_B_long_baseline['Y_avg'] = lambda_over_B_long_baseline[['Y_E', 'Y_W']].mean(axis=1)

lambda_over_B_short_baseline = pd.DataFrame.copy(lambda_over_B_N,deep=True)
lambda_over_B_short_baseline["Y_S"] = lambda_over_B_S["Y_S"]
lambda_over_B_short_baseline['Y_avg'] = lambda_over_B_short_baseline[['Y_N', 'Y_S']].mean(axis=1)


plt.gca().invert_yaxis()
plt.plot(lambda_over_B_long_baseline["X_E"],lambda_over_B_long_baseline["Y_E"], color="g", linestyle="--")
plt.plot(lambda_over_B_long_baseline["X_E"],lambda_over_B_long_baseline["Y_W"], color="g", linestyle="--")
plt.plot(lambda_over_B_long_baseline["X_E"],lambda_over_B_long_baseline["Y_avg"], color="g", linestyle="-")

plt.plot(lambda_over_B_short_baseline["X_N"],lambda_over_B_short_baseline["Y_N"], color="r", linestyle="--")
plt.plot(lambda_over_B_short_baseline["X_N"],lambda_over_B_short_baseline["Y_S"], color="r", linestyle="--")
plt.plot(lambda_over_B_short_baseline["X_N"],lambda_over_B_short_baseline["Y_avg"], color="r", linestyle="-")
plt.show()

#pd.concat([lambda_over_B_E, lambda_over_B_W], axis=1)
#lambda_over_B_short_baseline = pd.concat([lambda_over_B_N, lambda_over_B_S], axis=1)

#pd.mean(lambda_over_B_long_baseline)

# concatenate the averages
#df1 = pd.DataFrame([lambda_over_B_short_baseline["Y_avg"],lambda_over_B_long_baseline["Y_avg"]],axis=1)

#df_marks.mean(axis=0)
fig, ax = plt.subplots()
plt.gca().invert_yaxis()
#ax = plt.gca()
ax.set_xscale('log')
plt.tick_params(axis='x', which='minor', labelsize=17)
ax.xaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
#ax.xaxis.ticklabel_format(useOffset=False)
#print(len(lambda_over_B_long_baseline["X_E"]))
print(len(lambda_over_B_short_baseline["X_N"]))
print(len(lambda_over_B_short_baseline["Y_avg"]))
print(len(lambda_over_B_long_baseline["Y_avg"]))
ax.set_xlim([0.14,0.42])
plt.plot(lambda_over_B_long_baseline["X_E"][0:16],np.subtract(lambda_over_B_short_baseline["Y_avg"][0:16],lambda_over_B_long_baseline["Y_avg"][0:16]))
plt.plot([0.14,0.5],[0.,0.], linestyle="--")
plt.xscale('log')
plt.title("(Short baseline curve) - (long baseline curve)\nNB: THIS IS PROBABLY WRONG; NOT CHECKED YET")
plt.show()
#print(lambda_over_B_short_baseline)
