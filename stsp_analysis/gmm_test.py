'''
combine data from STSP output files

make longitude versus time plot, as before (for in-transit spots only)

try a Gaussian Mixture Model on this data

'''

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import mixture
from scipy import linalg
import itertools
from matplotlib.patches import Ellipse
import matplotlib as mpl

# from scipy import stats
# from matplotlib.mlab import griddata


mpl.rcParams['font.size'] = 16

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



tmid_nspt = np.repeat(tmid, nspt).reshape((len(tmid), nspt))

tlim = 1400.0
yes1 = np.where((in_trans.ravel() >= bump_lim) & (tmid_nspt.ravel() < tlim))

xo = tmid_nspt.ravel()[yes1]
yo = y1.ravel()[yes1]
zo = r1.ravel()[yes1]
data3d = np.squeeze(np.array([ [xo], [yo], [zo] ]))

X_train = data3d.T

#### just try straight copying an example, then swap out data
np.random.seed(0)
# fit a Gaussian Mixture Model with two components
clf = mixture.GMM(n_components=32, covariance_type='full')
clf.fit(X_train)
Y_ = clf.predict(X_train)


fig = plt.figure()
ax = fig.add_subplot(111)
color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm' ,'k','y'])

# find each group and how many points are in each
good, goodcts = np.unique(Y_, return_counts=True)

angle_out = np.zeros(clf.n_components)
for i in range(clf.n_components):
# for i in good[goodcts>5]:
    means = clf.means_[i, 0:2]

    # for GMM
    # covar = clf.covars_[i][0:2, 0:2]

    # for all modes of GMM
    covar = clf._get_covars()[i][0:2, 0:2]

    v, w, = linalg.eigh(covar)
    angle = np.arctan2(w[0][1], w[0][0]) * 180.0 / np.pi
    angle_out[i] = angle

    ell = Ellipse(means, v[0], v[1], angle)
    ax.add_artist(ell)
    ell.set_clip_box(ax.bbox)
    ell.set_facecolor('None')
    ell.set_fill(False)
    ell.set_color(color_iter.next())
    # ell.set_alpha(.25)

plt.xlim((np.min(tmid), np.max(tmid)))
plt.ylim((0,360))
plt.scatter(xo, yo, c='k', s=(zo / np.nanmax(r1)*20.)**2., alpha=0.6)
plt.show()

plt.figure()
h = plt.hist(angle_out)
plt.show()




plt.figure()
plt.scatter(xo, yo, c=Y_, s=(zo / np.nanmax(r1)*20.)**2., cmap=cm.Paired, alpha=0.6)
plt.xlabel('Time (BJD - 2454833 days)')
plt.ylabel('Longitude (deg)')
plt.title('Example GMM')
plt.xlim((np.min(tmid), np.max(tmid)))
plt.ylim((0,360))
cb = plt.colorbar()
cb.set_label('GMM component #')
plt.show()



# The general plot, replicate from IDL work
plt.figure()
for k in range(int(nspt)):
    yes = np.where((in_trans[:,k] >= bump_lim))
    plt.scatter(tmid[yes], y1[yes,k], cmap=cm.gnuplot2_r, c=(r1[yes,k]), alpha=0.6,
                s=(r1[yes,k] / np.nanmax(r1)*20.)**2.)
plt.xlim((np.min(tmid), np.max(tmid)))
plt.ylim((0,360))
plt.xlabel('Time (BJD - 2454833 days)')#, fontsize=18)
plt.ylabel('Longitude (deg)')#, fontsize=18)
plt.title('In-Transit Spots Only')#, fontsize=18)
cb = plt.colorbar()
cb.set_label('spot radius')#, fontsize=18)
plt.show()


#
# ntrials = 1
# ncomp = np.arange(1,40)
# bic = np.zeros_like(ncomp)
#
# plt.figure()
# for k in range(ntrials):
#
#     i=0
#     for n in ncomp:
#         np.random.seed(k+n)
#         clf = mixture.GMM(n_components=n, covariance_type='full')
#         clf.fit(X_train)
#         Y_ = clf.predict(X_train)
#         bic[i] = clf.bic(X_train)
#         i=i+1
#     plt.plot(ncomp,bic,'b',alpha=0.5)
# plt.xlabel('N components')
# plt.ylabel('BIC')
# plt.show()

