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
from scipy import stats
from sklearn import mixture

workingdir = '/astro/store/scratch/jrad/stsp/joe/'
actionL = True # has the Action=L rerun been done to make vis files?
bump_lim = 1 # number of epochs bump must exist for


# read in each parambest file, save as big structure
os.system('ls ' + workingdir + '*parambest*.txt > parambest.lis')
pbestfile = np.loadtxt('parambest.lis', dtype='string')

# read first file in to learn size
t = np.loadtxt(pbestfile[0], dtype='float', usecols=(0,), unpack=True, delimiter=' ')
nspt = (len(t) - 3.) / 2. / 3.

# variables to store spot properties (parambest files)
r1 = np.zeros((len(pbestfile), nspt))
x1 = np.zeros((len(pbestfile), nspt))
y1 = np.zeros((len(pbestfile), nspt))
chi = np.zeros_like(pbestfile)

# properties for each solution window (lcbest files)
in_trans = np.zeros((len(pbestfile), nspt))
tmid = np.arange(len(pbestfile)) # mid-transit time

for n in range(len(pbestfile)):
    t = np.loadtxt(pbestfile[n], dtype='float', usecols=(0,), unpack=True, delimiter=' ')
    np_l = nspt * 3 + 1
    chi[n] = t[np_l]

    # read in the lcbest file (or lcout)
    if actionL is True:
        tn,fn,en,mn,flg,x = np.loadtxt(pbestfile[n].replace('parambest', 'L_lcout'),
                                     dtype='float', unpack=True)
    else:
        tn,fn,en,mn = np.loadtxt(pbestfile[n].replace('parambest', 'lcbest'),
                                 dtype='float', unpack=True)
        flg = np.zeros_like(tn)
        x = np.zeros_like(tn)

    tmid[n] = np.median(tn)



    for i in range(int(nspt)):
        r1[n,i] = t[np_l + 1 + i*3.]
        x1[n,i] = t[np_l + 2 + i*3.]
        y1[n,i] = t[np_l + 3 + i*3.]

        k = np.mod(flg, 2)
        in_trans[n,i] = (k == 1.).sum()

        flg = (flg - k)/2.0


# follow scipy.stats tutorial for Kernel Density Estimator
yes1 = np.where((in_trans >= bump_lim))

tmid_nspt = np.repeat(tmid, nspt).reshape((len(tmid), nspt))

# data3d = np.squeeze(np.array([[tmid_nspt[yes1,:].ravel()],
#           [y1[yes1,:].ravel()],
#           [r1[yes1,:].ravel()]]))
data3d = np.squeeze(np.array([[tmid_nspt[yes1,:].ravel()],
          [y1[yes1,:].ravel()]]))

samples = data3d.T

gmix = mixture.GMM(n_components=5, covariance_type='full')
gmix.fit(samples)

colors = ['r' if i==0 else 'g' for i in gmix.predict(samples)]
ax = plt.gca()
ax.scatter(samples[:,0], samples[:,1], c=colors, alpha=0.6)#, s=(samples[:,2]/np.nanmax(samples[:,2])*20.)**2.)
plt.show()




# The general plot, replicate from IDL work
plt.figure()
for k in range(int(nspt)):
    yes = np.where((in_trans[:,k] >= bump_lim))
    plt.scatter(tmid[yes], y1[yes,k], cmap=cm.gnuplot2_r, c=(r1[yes,k]), alpha=0.6,
                s=(r1[yes,k] / np.nanmax(r1)*20.)**2.)
plt.xlim((np.min(tmid), np.max(tmid)))
plt.ylim((0,360))
plt.xlabel('Time (BJD - 2454833 days)')
plt.ylabel('Longitude (deg)')
cb = plt.colorbar()
cb.set_label('spot radius')
plt.title('In-Transit Spots Only')
plt.show()