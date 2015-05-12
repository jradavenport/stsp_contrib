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
# impact = str(params[5])
# incl_orb = str(params[6])
# Teff = str(params[7])
# sden = str(params[8])


# figure out number of spots by reading in first parambest file
os.system('ls ' + workingdir + '*parambest*.txt > parambest.lis')
pbestfile = np.loadtxt('parambest.lis', dtype='string')

# read first file in to learn size
tx = np.loadtxt(pbestfile[0], dtype='float', usecols=(0,), unpack=True, delimiter=' ')
nspt = (len(tx) - 3.) / 2. / 3.

chisq_tr = np.array([], dtype='float') # use np.append to add on to
tmid_tr = np.array([], dtype='float')

rad = np.empty(nspt, dtype='float') # use np.vstack (or other stack?) to add on to
lat = np.empty(nspt, dtype='float')
lon = np.empty(nspt, dtype='float')

for n in range(len(pbestfile)):
    # read in each lcbest file
    # NOTE: needs to have _L mode done, we need the vis flag!
    tn,fn,en,mn,flg = np.loadtxt(pbestfile[n].replace('parambest', 'L_lcout'),
                                 dtype='float', unpack=True, usecols=(0,1,2,3,4))
    np_l = nspt * 3 + 1
    i = range(int(nspt))

    # read in corresponding parambest file
    t = np.loadtxt(pbestfile[n], dtype='float', usecols=(0,), unpack=True, delimiter=' ')

    # step thru every transit in this lcbest file
    trans0 = int(np.round((tn[0] - float(t_0))/float(p_orb)))
    trans1 = int(np.round((tn[-1] - float(t_0))/float(p_orb)))
    for j in range(trans0, trans1):
        trans_j = float(t_0) + float(p_orb)*j
        intrans = np.where((tn >= trans_j - float(tdur)*0.5) &
                           (tn <= trans_j + float(tdur)*0.5))

        if len(intrans[0])>0:
            # for each transit, compute local chisq
            chisq_tr = np.append(chisq_tr, np.sum( ((mn[intrans] - fn[intrans]) / en[intrans])**2.0 ))

            tmid_tr = np.append(tmid_tr, np.mean(tn[intrans]))

            flg_tr = flg[intrans]
            r1 = np.zeros(nspt)
            x1 = np.zeros(nspt)
            y1 = np.zeros(nspt)
            in_trans = np.zeros(nspt)
            for i in range(int(nspt)):
                r1[i] = t[np_l + 1 + i*3.] # radius
                x1[i] = t[np_l + 2 + i*3.] # lat
                y1[i] = t[np_l + 3 + i*3.] # lon

                k = np.mod(flg_tr, 2)
                in_trans[i] = (k == 1.).sum()
                flg_tr = (flg_tr - k)/2.0

            bad = np.where((in_trans<1)) # find spots not in THIS transit
            if len(bad[0])>0:
                r1[bad] = -99 # give them bad values
                x1[bad] = -99
                y1[bad] = -99

            rad = np.vstack( (rad, r1) )
            lat = np.vstack( (lat, x1) )
            lon = np.vstack( (lon, y1) )

rad = rad[1:,:] # get rid of first empty row
lat = lat[1:,:]
lon = lon[1:,:]


tmid_out = np.array([], dtype='float')

rad_out = np.empty(nspt, dtype='float') # use np.vstack (or other stack?) to add on to
lat_out = np.empty(nspt, dtype='float')
lon_out = np.empty(nspt, dtype='float')

# once every lcbest file read, pass thru and compare windows in same transit
for j in range(int( (max(tmid_tr)-min(tmid_tr))/float(p_orb)+1 )):
    # the transit mid time
    trans_j = float(t_0) + float(p_orb)*j
    # find all transits that have same mid time, within some tolerance (0.1 days)
    xtr = np.where((np.abs(tmid_tr - trans_j)<0.1))
    if len(xtr[0])>0:
        best = np.argmin(chisq_tr[xtr])

        tmid_out = np.append(tmid_out, trans_j)
        rad_out = np.vstack( (rad_out, rad[xtr,:][:,best]) )
        lat_out = np.vstack( (lat_out, lat[xtr,:][:,best]) )
        lon_out = np.vstack( (lon_out, lon[xtr,:][:,best]) )

rad_out = rad_out[1:,:] # get rid of first empty row
lat_out = lat_out[1:,:]
lon_out = lon_out[1:,:]

plt.figure(figsize=(12,6))
for k in range(int(nspt)):
    yes = np.where((rad_out[:,k] > 0)) # pick spots not = -99
    plt.scatter(tmid_out[yes], lon_out[yes,k], alpha=0.6,
                cmap=cm.gnuplot2_r, c=(rad_out[yes,k]),
                s=(rad_out[yes,k] / np.nanmax(rad_out)*20.)**2.)
plt.xlim((np.min(tmid_out), np.max(tmid_out)))
plt.ylim((0,360))
plt.xlabel('Time (BJD - 2454833 days)')
plt.ylabel('Longitude (deg)')
plt.title('Best Transit Spots')
cb = plt.colorbar()
cb.set_label('spot radius')
plt.show()