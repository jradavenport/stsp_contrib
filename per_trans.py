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
import matplotlib.pyplot as plt
import matplotlib.cm as cm


