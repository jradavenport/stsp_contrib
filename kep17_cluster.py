'''
Do the DBSCAN clustering on STSP results
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


#--------- START OF SETUP ------------
# CONFIG THESE THINGS FOR EACH RUN

# which directory to run in? what settings?
workingdir = '/astro/store/scratch/jrad/stsp/n8s/'
actionL = True # has the Action=L rerun been done to make vis files?
bump_lim = 1 # number of epochs bump must exist for

# LC specific settings
per = 12.25817669188 # the rotation period used to fold this data and fed to STSP previous to this
tlim = 1600.0

# parameters for DBSCAN
cdist = 20.0 # max distance between points to be in cluster
cmin = 2 # min number of points to form a cluster


#--------- END OF SETUP ------------


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
        tn,fn,en,mn,flg = np.loadtxt(pbestfile[n].replace('parambest', 'L_lcout'),
                                     dtype='float', unpack=True)
    else:
        tn,fn,en,mn = np.loadtxt(pbestfile[n].replace('parambest', 'lcbest'),
                                 dtype='float', unpack=True)
        flg = np.zeros_like(tn)
        x = np.zeros_like(tn)

    tmid[n] = np.median(tn)

    for i in range(int(nspt)):
        r1[n,i] = t[np_l + 1 + i*3.] # radius
        x1[n,i] = t[np_l + 2 + i*3.] # lat
        y1[n,i] = t[np_l + 3 + i*3.] # lon

        k = np.mod(flg, 2)
        in_trans[n,i] = (k == 1.).sum()

        flg = (flg - k)/2.0

tmid_nspt = np.repeat(tmid, nspt).reshape((len(tmid), nspt))

yes1 = np.where((in_trans.ravel() >= bump_lim) & (tmid_nspt.ravel() < tlim))

xo = tmid_nspt.ravel()[yes1] # time
yo = y1.ravel()[yes1] # lon
zo = r1.ravel()[yes1] # rad
lo = x1.ravel()[yes1] # lat

data2d = np.squeeze(np.array([ [xo], [yo] ]))
X2d = data2d.T

data3d = np.squeeze(np.array([ [xo], [yo], [zo] ]))
X3d = data3d.T

data4d = np.squeeze(np.array([ [xo], [yo], [zo], [lo] ]))
X4d = data4d.T


# the data is now ready for clustering!!
###############################################

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
plt.title('In-Transit Spots Only')
cb = plt.colorbar()
cb.set_label('spot radius')
plt.show()
#####################


#-- follow DBSCAN example from:
# http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#example-cluster-plot-dbscan-py

# pick which version of the data to use
Xdbs = X3d # now do it in the 3D space

db = DBSCAN(eps=cdist, min_samples=cmin, algorithm='kd_tree').fit(Xdbs)

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
medlat = []
stdlat = []
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = Xdbs[class_member_mask & core_samples_mask]
    xy4 = X4d[class_member_mask & core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1], c=col,
            s=(xy[:, 2] / np.nanmax(r1)*20.)**2. )

    if (k != -1) and (len(xy[:,0]) > 2):
        coeff = np.polyfit(xy[:,0], xy[:,1], 1)
        xx = np.array( [min(xy[:,0]), max(xy[:,0])] )
        plt.plot(xx, np.polyval(coeff, xx), color='k', linewidth=4)

        slope.append( coeff[0]/360. ) # save the slope in phase units
        per2.append( per / (1.0 - (coeff[0] / 360. * per)) )
        medlat.append( np.median(xy4[:,3])-90. )
        stdlat.append( np.std(xy4[:,3]) )


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
plt.xlabel('Period (days)')
plt.show()


plt.figure()
plt.errorbar(per2, medlat,yerr=stdlat,fmt=None)
plt.scatter(per2, medlat, marker='o')
plt.xlabel('Period (days)')
plt.ylabel('Latitude (deg)')
plt.show()

# make these other plots
# 1. (cluster duration, peak spot size) scatter plot for all clusters
# 2. (time, spot radius) overlap time series for all clusters
#

# Compute the differential rotation (k) parameter