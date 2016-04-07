# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
from glob import glob


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:
        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)

    see also:
        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter
    """
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(), s, mode='valid')
    return y[(window_len/2-1):-(window_len/2)]

#############    SET UP PARAMETERS
prefix = 'run0'
BJDREF = 2454833.

FLATMODE = True

# !!- Make sure to include / at the end -!!
basedir = '/home/davenpj3/stsp/kepler17/'
workdir = basedir + prefix + '/'
datadir = basedir + 'k17_single_transits/'

shfile = workdir + 'runL_'+prefix+'.sh'

# location of the STSP code
stsp_ver = '/home/davenpj3/stsp/STSP/stsp_L'

#  SET UP MCMC & FITTING PARAMETERS
nspots = 3
ascale = '2.5' # the "jump" parameter
npop = '100' # walkers

nsteps = '30000' # do more steps for resume

# the 4th order limb darkening array
limb_array = '0.59984  -0.165775  0.6876732  -0.349944' # for Kep 17

# the parameter file detailing the system
pfile = '/home/davenpj3/stsp/kepler17/kep17.params'

params = np.loadtxt(pfile, unpack=True, skiprows=4, usecols=[0])
p_orb = str(params[0])
t_0 = str(params[1]) # - BJDREF)
tdur = str(params[2])
p_rot = str(params[3])
Rp_Rs = str(params[4]**2.)
impact = str(params[5])
incl_orb = str(params[6])
Teff = str(params[7])
sden = str(params[8])

#############    END OF PARAMETER SETUP



# 2222222222222222 create CONDOR .cfg file
f2 = open(shfile,'w')
f2.write("#!/bin/sh \n")


# main loop for .in file writing of each time window & #spots
for datafile in glob(datadir + 'transit*.txt'):
    t,flux,err = np.loadtxt(datafile, unpack=True)

    dstart = min(t)
    ddur = max(t) - min(t)

    # name of .in file to use for this time window:
    basename = datafile.split(datadir)[1] + '_' + prefix

    file = basename + '1L.in'

    f = open(workdir + file, 'w')

    f.write('#PLANET PROPERTIES\n')
    f.write('1\n')            #-- # planets
    f.write(t_0 + '\n')       #-- T0  (middle of first transit) in days.
    f.write(p_orb + '\n')     #-- Planet Period (days)
    f.write(Rp_Rs + '\n')     #-- (Rp/Rs)^2
    f.write(tdur + '\n')      #-- Duration (days)
    f.write(impact + '\n')    #-- Impact parameter (0=planet cross equator)
    f.write(incl_orb + '\n')  #-- Incl angle of orbit (90deg=planet cross over equator)
    f.write('0\n' )           #-- sky projected obliquity ("lambda")
    f.write('0.0\n')          #-- ecosw (leave 0)
    f.write('0.0\n')          #-- esinw (leave 0)

    f.write('#STAR PROPERTIES\n')
    f.write(sden + '\n')          #-- stellar density ( this from Leslie )
    f.write(p_rot + '\n')          #-- rotation period
    f.write(Teff + '\n')           #-- Teff
    f.write('0.0\n')               #-- Metallicity (not used)
    f.write('0.0\n')               #-- inclination
    f.write(limb_array + '\n') #-- limb darkening (from Leslie's calculation)
    f.write('40\n')               #-- # rings for limb darkening approx.

    f.write('#SPOT PROPERTIES\n')
    f.write(str(nspots) + '\n')       #-- # of spots
    f.write('0.7\n')               #-- contrast

    f.write('# FITTING properties\n')
    f.write(datafile + '\n')     #-- the datafile
    f.write(str(dstart) + '\n')    #-- tstart
    f.write(str(ddur) + '\n')      #-- duration (days)

    maxwindow = 1.0 # assume this for the "flat" case

    f.write(str(maxwindow)+'\n')   #-- real max of LC, 0=use downfrommax
    f.write('1\n') #-- is LC flattened to zero outside of transit? (1=yes,0=no)

    f.write('#ACTION\n')           #   what to do w/ STSP?
    f.write('L\n')

    f.write(basename + '1_parambest.txt \n')

    f.close()

    # put entry in to shell script file for this window
    f2.write(stsp_ver + ' ' + file + ' \n')


f2.write(' \n')
f2.close()



# fix permissions
os.system("chmod 777 "+shfile)


print("New .in files created for -L mode generation, run shell script file "+shfile+" to compute -L mode outputs")
