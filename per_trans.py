'''
for each transit:
- find any detected bumps within +/- 1 model
- find which model fits that bump the best
- use the best model to determine the position of the spot at that transit time
- find any missed bumps
    - pick best overall model, look for residuals
- save big structure or file of spot (position, rad, time)
'''

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm



#--------- START OF SETUP ------------
# CONFIG THESE THINGS FOR EACH RUN
#-- which directory to run in? what settings?
fname = 'k17'
workingdir = '/astro/store/scratch/jrad/stsp/n8s/' # kepler17
dropboxdir = '/astro/users/jrad/Dropbox/python/kepler17/' # kepler17
BJDREF = 2454833. # kepler17

# fname = 'joe'
# workingdir = '/astro/store/scratch/jrad/stsp/joe/' # joe model
# dropboxdir = '/astro/users/jrad/Dropbox/python/kepler17/llama_model/' # joe model
# BJDREF = 0.  # Joe model

#--------- END OF SETUP ------------



#-- read in original .params file, get (t_0, p_orb) to find each transit
pfile = dropboxdir + 'kep17.params'

params = np.loadtxt(pfile,unpack=True,skiprows=4,usecols=[0])
p_orb = str(params[0])
t_0 = str(params[1] - BJDREF)
tdur = str(params[2])
p_rot = str(params[3])
Rp_Rs = str(params[4]**2.)
impact = str(params[5])
incl_orb = str(params[6])
Teff = str(params[7])
sden = str(params[8])


# figure out number of spots by reading in first parambest file
os.system('ls ' + workingdir + '*parambest*.txt > parambest.lis')
pbestfile = np.loadtxt('parambest.lis', dtype='string')

# read first file in to learn size
tx = np.loadtxt(pbestfile[0], dtype='float', usecols=(0,), unpack=True, delimiter=' ')
nspt = (len(tx) - 3.) / 2. / 3.


for n in range(len(pbestfile)):
    # read in each lcbest file
    # NOTE: needs to have _L mode done, we need the vis flag!
    tn,fn,en,mn,flg = np.loadtxt(pbestfile[n].replace('parambest', 'L_lcout'),
                                 dtype='float', unpack=True, usecols=(0,1,2,3,4))
    # read in corresponding parambest file
    t = np.loadtxt(pbestfile[n], dtype='float', usecols=(0,), unpack=True, delimiter=' ')

# step thru every transit in this lcbest file
# for each transit, compute local chisq
# save grid of numbers for each transit:
#   t_mid, chisq, (rad, lat, lon) for each bump present - pad up to 8 spots
# once every lcbest file read, transit stored,
#   do a pass thru grid of data, if same transit done in mult.
#   lcbest files, pick best, remove others!


# read in each parambest file, save as big structure

