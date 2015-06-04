'''
Do the DBSCAN clustering on STSP results
'''

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from sklearn.cluster import DBSCAN
from sklearn import linear_model, datasets

mpl.rcParams['font.size'] = 16


#--------- START OF SETUP ------------
# CONFIG THESE THINGS FOR EACH RUN

# fname = 'k17'
fname = 'joe'



if (fname == 'joe'):
    workingdir = '/astro/store/scratch/jrad/stsp/joe/' # joe model
    per = 10.0
    tlim = 1400.0 #- Joe model
    phz_max = 180.0

if (fname == 'k17'):
    workingdir = '/astro/store/scratch/jrad/stsp/n8s/' # kepler17
    per = 12.25817669188 # the rotation period used to fold this data and fed to STSP previous to this
    tlim = 1600.0 #- Kepler 17
    phz_max = 360.0

phz_min = phz_max - 360.

# parameters for DBSCAN
cdist = 15.0 # max distance between points to be in cluster
cmin = 3 # min number of points to form a cluster
actionL = True # has the Action=L rerun been done to make vis files?
bump_lim = 1 # number of epochs bump must exist for


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
                                     dtype='float', unpack=True, usecols=(0,1,2,3,4))
    else:
        tn,fn,en,mn = np.loadtxt(pbestfile[n].replace('parambest', 'lcbest'),
                                 dtype='float', unpack=True, usecols=(0,1,2,3))
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

y1[np.where((y1 > phz_max))] = y1[np.where((y1 > phz_max))] - 360.

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
plt.figure(figsize=(12,6))
for k in range(int(nspt)):
    yes = np.where((in_trans[:,k] >= bump_lim))
    plt.scatter(tmid[yes], y1[yes,k], cmap=cm.gnuplot2_r, c=(r1[yes,k]), alpha=0.6,
                s=(r1[yes,k] / np.nanmax(r1)*20.)**2.)
plt.xlim((np.min(tmid), np.max(tmid)))
plt.ylim((phz_min, phz_max))
plt.xlabel('Time (BJD - 2454833 days)')
plt.ylabel('Longitude (deg)')
plt.title('In-Transit Spots Only')
cb = plt.colorbar()
cb.set_label('spot radius')
plt.savefig('/astro/users/jrad/Dropbox/research_projects/gj1243_spots/'+fname+'_lon_v_time.pdf', dpi=250, bbox_inches='tight')
plt.close()
# plt.show()
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

# parameters to measure within each cluster
slope = []
per2 = []
medlat = []
stdlat = []
cdur = [] # duration of cluster (regular dur? folding time?)
cpeak = [] # cluster peak size

plt.figure(figsize=(10,6))
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
        # if cluster is good, measure parameters
        coeff = np.polyfit(xy[:,0], xy[:,1], 1)
        xx = np.array( [min(xy[:,0]), max(xy[:,0])] )
        plt.plot(xx, np.polyval(coeff, xx), color='k', linewidth=4)

        slope.append( coeff[0]/360. ) # save the slope in phase units
        per2.append( per / (1.0 - (coeff[0] / 360. * per)) )
        medlat.append( np.median(xy4[:,3])-90. )
        stdlat.append( np.std(xy4[:,3]) )

        cdur.append( np.max(xy[:,0]) - np.min(xy[:,0]) )
        cpeak.append( np.max(xy[:,2]) )


    xy = Xdbs[class_member_mask & ~core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1], c=col,
             alpha=0.25, s=(xy[:, 2] / np.nanmax(r1)*20.)**2.)

plt.title('DBSCAN: Estimated number of clusters: %d' % n_clusters_)
plt.xlabel('Time (BJD - 2454833 days)')
plt.ylabel('Longitude (deg)')
plt.xlim((np.min(tmid), np.max(tmid)))
plt.ylim((phz_min, phz_max))
# cb = plt.colorbar() #oops, this isn't mappable
# cb.set_label('cluster number')
plt.savefig('/astro/users/jrad/Dropbox/research_projects/gj1243_spots/'+fname+'_lon_v_time_cluster.pdf', dpi=250, bbox_inches='tight')
plt.close()
# plt.show()


plt.figure()
h = plt.hist(per2)
plt.xlabel('Period (days)')
plt.ylabel('Number of Clusters')
plt.savefig('/astro/users/jrad/Dropbox/research_projects/gj1243_spots/'+fname+'_per_hist.pdf', dpi=250, bbox_inches='tight')
plt.close()
# plt.show()

print(np.percentile(per2, [5,95]))




plt.figure()
plt.errorbar(per2, medlat,yerr=stdlat,fmt=None)
plt.scatter(per2, medlat, marker='o')

if (fname == 'joe'):
    #overplot the solar law
    lat = np.arange(-40,40)
    k = 1.0 # delta Omeag / Omega
    per_lat = per / (1 - k*(np.sin(lat/180.*np.pi)**2.0))
    plt.plot(per_lat, lat, 'k')

    k = [0.2] # delta Omeag / Omega for Sun
    per_lat = per / (1 - k*(np.sin(lat/180.*np.pi)**2.0))
    plt.plot(per_lat, lat, 'red')

    plt.ylim((-15, 15))
    plt.xlim((9.6, 10.4))

plt.xlabel('DBSCAN Cluster Period (days)')
plt.ylabel('Median Cluster Latitude (deg)')
plt.savefig('/astro/users/jrad/Dropbox/research_projects/gj1243_spots/'+fname+'_per_v_lat.pdf', dpi=250, bbox_inches='tight')
plt.close()
# plt.show()


#-- make these other plots

# 1. (cluster duration, peak spot size) scatter plot for all clusters

# fit a line to this! Use robust random sampling w/ outliers
model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
model_ransac.fit(np.array(cdur)[:,np.newaxis], np.array(cpeak)[:,np.newaxis])
line_y_ransac = model_ransac.predict(np.array(cdur)[:,np.newaxis])

plt.figure()
plt.scatter(cdur, cpeak, marker='d',color='k')
plt.plot(cdur, line_y_ransac, '-k')
plt.xlabel('Cluster Duration (days)')
plt.ylabel('Max Spot Radius')
plt.savefig('/astro/users/jrad/Dropbox/research_projects/gj1243_spots/'+fname+'_dur_v_rad.pdf', dpi=250, bbox_inches='tight')
plt.close()
# plt.show()


# now same figure in area units
R_sun = 6.955e10 # cm
R_star = R_sun # * 1.02
microHem = 3e16 # cm sq


plt.figure()
plt.scatter(cdur, (np.pi * (np.array(cpeak,dtype='float')*R_star)**2.0) / microHem, marker='d',color='k')
# plt.plot(cdur, line_y_ransac, '-k')
plt.xlabel('Cluster Duration (days)')
plt.ylabel('Max Spot Area ($\mu$Hem)')
plt.savefig('/astro/users/jrad/Dropbox/research_projects/gj1243_spots/'+fname+'_dur_v_area.pdf', dpi=250, bbox_inches='tight')
plt.close()



# 2. (time, spot radius) overlap time series for all clusters
plt.figure()
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'
    class_member_mask = (labels == k)
    xy = Xdbs[class_member_mask & core_samples_mask]
    if (k != -1) and (len(xy[:,0]) > 2):
        tmax = xy[:,0][np.argmax(xy[:,2])]
        plt.plot(xy[:,0]-tmax, xy[:,2], color=col)

plt.xlabel('Time (days)')
plt.ylabel('Radius')
plt.savefig('/astro/users/jrad/Dropbox/research_projects/gj1243_spots/'+fname+'_time_v_rad.pdf', dpi=250, bbox_inches='tight')
plt.close()
# plt.show()



# now same figure in area units
R_sun = 6.955e10 # cm
R_star = R_sun # * 1.02
microHem = 3e16 # cm sq

plt.figure(figsize=(10,6))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'
    class_member_mask = (labels == k)
    xy = Xdbs[class_member_mask & core_samples_mask]
    if (k != -1) and (len(xy[:,0]) > 5) and \
            (np.max(xy[:,2])>0.1) and (np.max(xy[:,2])/np.min(xy[:,2]) > 1.5):
        tmax = xy[:,0][np.argmax(xy[:,2])]
        area = (np.pi * (xy[:,2]*R_star)**2.0) / microHem

        plt.plot(xy[:,0]-tmax, area, color=col)
        plt.scatter(xy[:,0]-tmax, area, c=col)

sunT = np.arange(0.,50.)
maxA = 17000.0
sunA = np.zeros_like(sunT) + maxA
for i in range(len(sunT)-1):
    dA = 24. + 0.116*sunA[i]
    sunA[i+1] = sunA[i] - dA
plt.plot(sunT, sunA, 'k')

plt.xlabel('Time (days)')
plt.ylabel('Area ($\mu$Hem)')
plt.xlim((-10,45))
plt.ylim((0,2e4))
plt.savefig('/astro/users/jrad/Dropbox/research_projects/gj1243_spots/'+fname+'_time_v_area.pdf', dpi=250, bbox_inches='tight')
plt.close()
# plt.show()

