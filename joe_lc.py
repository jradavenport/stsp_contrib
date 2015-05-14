'''
Make figure(s) introducing Joe Llama's model light curve he has provide us

can be run on the laptop, just run in the dropbox dir
'''


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 16



# dropboxdir = '/astro/users/jrad/Dropbox/python/kepler17/llama_model/'
dropboxdir = '/Users/james/Dropbox/python/kepler17/llama_model/'

prefix = 'joe'
BJDREF = 0.

# the parameter file detailing the system
pfile = 'llama.params'

params = np.loadtxt(dropboxdir + pfile,unpack=True,skiprows=4,usecols=[0])
p_orb = str(params[0])
t_0 = str(params[1] - BJDREF)
tdur = str(params[2])
p_rot = str(params[3])
Rp_Rs = str(params[4]**2.)
impact = str(params[5])
incl_orb = str(params[6])
Teff = str(params[7])
sden = str(params[8])


# where is the original data file to analyze?
#-- for dealing w/ Joe's model data:
datafile = 'llama_lightcurve_raw.txt'
tbad,ff0 = np.loadtxt(dropboxdir + datafile, unpack=True, usecols=(0,1), dtype=('string','float'))
escale = 0.0002 # value i arrived at "a posteriori"
ee0 = np.random.normal(loc=0,scale=escale,size=np.size(ff0))
# remake time array
dt = 5. / 60. / 24. # 5 min cadence
tt0 = np.arange(np.size(ff0))*dt


toff = float(tdur) / float(p_orb) # the transit duration in % (phase) units

# down-sample the out-of-transit data
# define the region +/- 1 transit duration for in-transit
OOT = np.where( ((((tt0-float(t_0)) % float(p_orb))/float(p_orb)) > toff) &
                ((((tt0-float(t_0)) % float(p_orb))/float(p_orb)) < 1-toff) )

ITT = np.where( ((((tt0-float(t_0)) % float(p_orb))/float(p_orb)) <= toff) |
                ((((tt0-float(t_0)) % float(p_orb))/float(p_orb)) >= 1-toff) )


# the factor to down-sample by
Nds = 30 # for Kepler 17
# Nds = 10 # for Joe's data

# down sample out-of-transit, grab every N'th data point
OOT_ds = OOT[0][np.arange(0,np.size(OOT), Nds)]

# use in- and out-of-transit data
idx = np.concatenate((ITT[0][0:], OOT_ds))
idx.sort()

# these arrays are the final, down-sampled data
tt = tt0[idx]
# ff = ff0[idx]
# ee = ee0[idx]
#-- for Joe's model, add noise
ff = np.add( np.array(ff0[idx], dtype='float'), ee0[idx] )
ffraw = np.array(ff0[idx], dtype='float')
ee = np.ones_like(ff) * escale


plt.figure(figsize=(13,6))
plt.scatter(tt, ffraw, marker='o')
plt.plot(tt, ffraw)
plt.xlim((855,915))
plt.ylim((0.97, 1.0))
plt.xlabel('Time (days)')
plt.ylabel('Relative Flux')
plt.savefig('/Users/james/Dropbox/research_projects/gj1243_spots/joe_lc_wide1.png', dpi=250)
plt.show()


plt.figure(figsize=(13,6))
plt.scatter(tt, ffraw, marker='o')
plt.plot(tt, ffraw)
plt.xlim((143.1,144.1))
plt.ylim((0.97,0.985))
plt.xlabel('Time (days)')
plt.ylabel('Relative Flux')
plt.savefig('/Users/james/Dropbox/research_projects/gj1243_spots/joe_lc_trans.png', dpi=250)
plt.show()
