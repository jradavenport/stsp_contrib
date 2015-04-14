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

workingdir = '/astro/store/scratch/jrad/stsp/n8s/'

# read in each parambest file, save as big structure
os.system('ls ' + workingdir + '*parambest*.txt > parambest.lis')
pbestfile = np.loadtxt('parambest.lis', dtype='string')

# read first file in to learn size
t = np.loadtxt(pbestfile[0], dtype='float', usecols=(0,), unpack=True, delimiter=' ')
nspt = (len(t) - 3.) / 2. / 3.

# variables to store spot properties
r1 = np.zeros((len(pbestfile), nspt))
x1 = np.zeros((len(pbestfile), nspt))
y1 = np.zeros((len(pbestfile), nspt))
in_trans = np.zeros((len(pbestfile), nspt))

# properties for each solution
chi = np.zeros_like(pbestfile) # chisq
tmid = np.arange(len(pbestfile)) # mid-transit time

for n in range(len(pbestfile)):
   t = np.loadtxt(pbestfile[n], dtype='float', usecols=(0,), unpack=True, delimiter=' ')
   np_l = nspt * 3 + 1
   chi[n] = t[np_l]

   for k in range(int(nspt)):
      r1[n,k] = t[np_l + 1 + k*3.]
      x1[n,k] = t[np_l + 2 + k*3.]
      y1[n,k] = t[np_l + 3 + k*3.]

# The general plot, replicate from IDL work
plt.figure()
for k in range(int(nspt)):
    plt.scatter(tmid, y1[:,k], cmap=cm.RdBu, c=r1[:,k], alpha=0.6,
                s=(r1[:,k]/np.nanmax(r1)*20.)**2.)
plt.xlim((np.min(tmid), np.max(tmid)))
plt.ylim((0,360))
plt.xlabel('Time')
plt.ylabel('Lon')
plt.show()

# read in each lcbest file, then step thru each transit