# -*- coding: utf-8 -*-

import numpy as np
import os


#############
#    SET UP PARAMETERS

# where the results are stored
# !!- Make sure to include / at the end -!!
workdir = '/astro/store/scratch/jrad/stsp/'
dropboxdir = '/astro/users/jrad/Dropbox/python/stsp_contrib/'


# the files to create...
shellscript = dropboxdir + 'pylaunch_cleaner.csh'
cfgfile     = dropboxdir + 'condor_cleaner.cfg'
stsprunner  = dropboxdir + 'python_cleaner.py'

pyversion = "/astro/apps6/anaconda/bin/python"


Nclean = 250 # number of jobs to queue up
#############



# 2222222222222222 create CONDOR .cfg file
f2 = open(cfgfile,'w')
f2.write('Notification = never \n')
f2.write('Executable = '+shellscript+' \n')
f2.write('Initialdir = ' + workdir + '\n')
f2.write('Universe = vanilla \n')
f2.write(' \n')
f2.write('Log = '+workdir+'log.txt \n')
f2.write('Error = '+workdir+'err.txt \n')
f2.write('Output = '+workdir+'out.txt \n')
f2.write(' \n')


# main loop for .in file writing of each time window & #spots
for n in range(0,Nclean):
    namen = format(int(n+1),"03d") # this is pointless but whatever

    # put entry in to CONDOR .cfg file for this job
    #f2.write('Arguments = ' + namen + ' \n')
    f2.write('Queue \n')
    
f2.write(' \n')
f2.close()



# 33333333333333333
# create the very simple PYTHON-launching shell script
f3 = open(shellscript,'w')
f3.write("#!/bin/bash \n")
f3.write(pyversion + " " + stsprunner + " \n")
f3.close()





clean_script = '''
import os
import time

time.sleep(10)

if os.path.isdir('/local/tmp/jrad/stsp_tmp'):
    time.sleep(1)
    os.system('chmod 755 /local/tmp/jrad/')
    try:
        os.system('rm -rf /local/tmp/jrad/stsp_tmp')
    except OSError:
        pass
time.sleep(1)
print 'done cleaning'

'''

# 444444444444444  generate the STSP-running script
f4 = open(stsprunner,'w')

f4.write(clean_script)

f4.close()


# fix permissions
os.system("chmod 777 "+stsprunner)
os.system("chmod 777 "+shellscript)
os.system("chmod 777 "+cfgfile)

# how to launch CONDOR jobs!
#  but actually, wait on that... do it manually. first test a few .in files
### os.system("condor_submit "+cfgfile)
