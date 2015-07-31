'''
- Go through all the _L_lcout.txt files, each transit, collect bumps
- for each bump, measure it's height relative to the overall flux max (median?)
'''
import numpy as np
import os
import matplotlib.pyplot as plt


prefix = 'n8s'
BJDREF = 2454833.


# where the results are stored
# !!- Make sure to include / at the end -!!
workingdir = '/astro/store/scratch/jrad/stsp/'+prefix+'/'
dropboxdir = '/astro/users/jrad/Dropbox/python/kepler17/'

pfile='kep17.params'

# where is the original data file to analyze?
datafile = 'kepler17_whole.dat'
tt0,ff0,ee0 = np.loadtxt(dropboxdir + datafile,unpack=True)

medflux = np.median(ff0)

#-- read in original .params file, get (t_0, p_orb) to find each transit
    # pfile = 'kep17.params'

params = np.loadtxt(dropboxdir + pfile, unpack=True, skiprows=4,usecols=[0])
p_orb = str(params[0])
t_0 = str(params[1] - BJDREF)
tdur = str(params[2])


# figure out number of spots by reading in first parambest file
os.system('ls ' + workingdir + '*parambest*.txt > parambest.lis')
pbestfile = np.loadtxt('parambest.lis', dtype='string')

# read first file in to learn size
tx = np.loadtxt(pbestfile[0], dtype='float', usecols=(0,), unpack=True, delimiter=' ')
nspt = (len(tx) - 3.) / 2. / 3.

# make list of lcout files to read and work thru
os.system('ls ' + workingdir + '*_L_lcout*.txt > lcout.lis')
lcoutfile = np.loadtxt('lcout.lis', dtype='string')


minbump = 1e-5
maxbump = 1e-2
numstep = 250

# bump_amp = np.logspace(np.log10(minbump), np.log10(maxbump), numstep)
bump_amp = np.linspace(minbump, maxbump, numstep)
bump_hist = np.zeros_like(bump_amp)
num_trans = 0

for n in range(len(lcoutfile)):
    tn,fn,en,mn,flg = np.loadtxt(lcoutfile[n],dtype='float',
                                 unpack=True, usecols=(0,1,2,3,4))

    # step thru every transit in this lcbest file
    trans0 = int(np.round((tn[0] - float(t_0))/float(p_orb)))
    trans1 = int(np.round((tn[-1] - float(t_0))/float(p_orb)))
    for j in range(trans0, trans1):
        trans_j = float(t_0) + float(p_orb)*j
        intrans = np.where((tn >= trans_j - float(tdur)*0.5) &
                           (tn <= trans_j + float(tdur)*0.5))

        if (len(intrans[0]) > 0):
            num_trans = num_trans + 1

            # in this transit, look for any bumps (flg[intrans] > 0)
            # b_it = np.where((flg[intrans] > 0))

            flg_tr = flg[intrans]

            for i in range(int(nspt)):
                # go thru each possible spot, if has bump then count it
                k = np.mod(flg_tr, 2)
                N_in_trans = (k == 1.).sum()
                flg_tr = (flg_tr - k)/2.0

                # for each bump, measure height (fit line around, subtract)
                if (N_in_trans > 0):
                    bi = np.where((k == 1))[0] # find all points where this bump is
                    # continuum regions immediately before/after bump
                    c1 = mn[intrans[0][bi][0] - 1]
                    c2 = mn[intrans[0][bi][-1] + 1]
                    t1 = tn[intrans[0][bi][0] - 1]
                    t2 = tn[intrans[0][bi][-1] + 1]
                    # compute linear slope between (crude)
                    m = (c1-c2) / (t1-t2)
                    b = c1 - m*t1
                    fit = [m,b]
                    # subtract linear slope from bump
                    flat = mn[intrans][bi] - np.polyval(fit,tn[intrans][bi])
                    # calculate highest point in bump
                    pk = max(flat) / medflux
                    # print(j,i,pk)

                    # find bump_amp bin that is closest
                    cl = np.argmin(np.abs(bump_amp - pk))
                    # then put bump in to respective bump_hist bin
                    bump_hist[cl] = bump_hist[cl] + 1


plt.figure()
plt.plot(bump_amp, bump_hist / num_trans)
plt.yscale('log')
plt.xscale('log')
plt.xlim((1e-4,1e-2))
plt.ylim((5e-4,1))
# plt.semilogy(bump_amp, bump_hist / num_trans)
plt.ylabel('Spots per Transit')
plt.xlabel('Bump Amplitude / Median Flux')
plt.show()


## answer the question: what is the odds of seeing a bump this big or bigger?
## this is not right...
# plt.figure()
# plt.plot(bump_amp[::-1], np.cumsum(bump_hist[::-1]/num_trans) / np.sum(bump_hist[::-1]/num_trans))
# plt.yscale('log')
# plt.xscale('log')
# plt.show()
