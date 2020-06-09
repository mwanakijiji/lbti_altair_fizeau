#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This reads in Jordan Stone's LEECH contrast curve for Altair for comparison


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[6]:


# LEECH data
contrast_leech = pd.read_csv("data/HIP97649_LEECH.txt")
radii_leech = pd.read_csv("data/radii_LEECH.txt")

# our Fizeau data
data_fizeau =  pd.read_csv("data/modern_contrast_curve.csv")


# In[7]:


data_fizeau.keys()


# In[15]:


plt.clf()
plt.plot(radii,contrast,
         label="Altair, LEECH\n(one telescope,\nt_int = 2873 sec.,\n83 deg rot,\nseeing 0.9)") # info from Table 4 in Stone+ 2018
plt.plot(data_fizeau["rad_asec"],data_fizeau["del_m_modern"],
         label="Altair, us (Fizeau)\n(\nt_int = TBD sec.,\n83 deg rot,\nseeing ~0.9-1.4)") 
plt.xlabel("Radius (asec)")
plt.ylabel("$\Delta m$")
plt.legend()
plt.gca().invert_yaxis()
plt.savefig("junk_cc_comparison.pdf")


# In[11]:


data_fizeau["F"]


# In[12]:


data_fizeau["rad_fwhm"]


# In[ ]:




