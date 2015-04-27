'''
Do the same as in the GMM test code, but try different clustering approaches

once clusters are robust, then fit lines to data within each cluster!
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
from sklearn.cluster import DBSCAN
from sklearn import metrics



mpl.rcParams['font.size'] = 16

workingdir = '/astro/store/scratch/jrad/stsp/joe/'
actionL = True # has the Action=L rerun been done to make vis files?
bump_lim = 1 # number of epochs bump must exist for


per = 10.0 # the rotation period used to fold this data and fed to STSP previous to this

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

data2d = np.squeeze(np.array([ [xo], [yo] ]))
X2d = data2d.T

# the data is now ready for clustering!!
###############################################

# The general plot, replicate from IDL work
# plt.figure()
# for k in range(int(nspt)):
#     yes = np.where((in_trans[:,k] >= bump_lim))
#     plt.scatter(tmid[yes], y1[yes,k], cmap=cm.gnuplot2_r, c=(r1[yes,k]), alpha=0.6,
#                 s=(r1[yes,k] / np.nanmax(r1)*20.)**2.)
# plt.xlim((np.min(tmid), np.max(tmid)))
# plt.ylim((0,360))
# plt.xlabel('Time (BJD - 2454833 days)')
# plt.ylabel('Longitude (deg)')
# plt.title('In-Transit Spots Only')
# cb = plt.colorbar()
# cb.set_label('spot radius')
# plt.show()
#####################

# now try different clustering approaches


#-- GMM
np.random.seed(0)
clf = mixture.GMM(n_components=32, covariance_type='full')
clf.fit(X_train)
Y_ = clf.predict(X_train)

# make plot of all the ellipses
# fig = plt.figure()
# ax = fig.add_subplot(111)
# color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm' ,'k','y'])
# angle_out = np.zeros(clf.n_components)
# for i in range(clf.n_components):
#     means = clf.means_[i, 0:2]
#     covar = clf._get_covars()[i][0:2, 0:2]
#
#     v, w, = linalg.eigh(covar)
#     angle = np.arctan2(w[0][1], w[0][0]) * 180.0 / np.pi
#     angle_out[i] = angle
#
#     ell = Ellipse(means, v[0], v[1], angle)
#     ax.add_artist(ell)
#     ell.set_clip_box(ax.bbox)
#     ell.set_facecolor('None')
#     ell.set_fill(False)
#     ell.set_color(color_iter.next())
#
# plt.xlim((np.min(tmid), np.max(tmid)))
# plt.ylim((0,360))
# plt.scatter(xo, yo, c='k', s=(zo / np.nanmax(r1)*20.)**2., alpha=0.6)
# plt.show()

# make standard plot, color by cluster
# plt.figure()
# plt.scatter(xo, yo, c=Y_, s=(zo / np.nanmax(r1)*20.)**2., cmap=cm.Paired, alpha=0.6)
# plt.xlabel('Time (BJD - 2454833 days)')
# plt.ylabel('Longitude (deg)')
# plt.title('Example GMM')
# plt.xlim((np.min(tmid), np.max(tmid)))
# plt.ylim((0,360))
# cb = plt.colorbar()
# cb.set_label('GMM component #')
# plt.show()


#-- follow DBSCAN example from:
# http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#example-cluster-plot-dbscan-py

#Xdbs = X2d # this now works in 2D
Xdbs = X_train # now do it in the 3D space

db = DBSCAN(eps=10, min_samples=2, algorithm='kd_tree').fit(Xdbs)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = cm.Paired( np.linspace(0, 1, len(unique_labels)) )

slope = []
per2 = []
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = Xdbs[class_member_mask & core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1], c=col,
            s=(xy[:, 2] / np.nanmax(r1)*20.)**2. )
    if (k != -1) and (len(xy[:,0]) > 4):
        coeff = np.polyfit(xy[:,0], xy[:,1], 1)
        xx = np.array( [min(xy[:,0]), max(xy[:,0])] )
        plt.plot(xx, np.polyval(coeff, xx), color='k', linewidth=4)

        slope.append(coeff[0]/360.) # save the slope in phase units
        per2.append( per / (1.0 - (coeff[0] / 360. * per)) )


    xy = Xdbs[class_member_mask & ~core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1], c=col,
             alpha=0.25, s=(xy[:, 2] / np.nanmax(r1)*20.)**2.)

plt.title('DBSCAN: Estimated number of clusters: %d' % n_clusters_)
plt.xlabel('Time (BJD - 2454833 days)')
plt.ylabel('Longitude (deg)')
plt.xlim((np.min(tmid), np.max(tmid)))
plt.ylim((0,360))
plt.show()


plt.figure()
h = plt.hist(per2)
plt.show()

