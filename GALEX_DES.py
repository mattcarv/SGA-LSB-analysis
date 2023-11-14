import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from scipy import stats
from scipy.stats import skewnorm
from scipy.optimize import curve_fit
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
plt.rcParams["figure.figsize"] = [10, 8]
plt.rcParams.update({'font.size': 18})

file = Table.read('C:/Users/mathe/Downloads/DxGxS.fits')
file.keep_columns(['SGA_ID_1', 'GALAXY', 'RA_LEDA', 'DEC_LEDA',
                   'MORPHTYPE', 'PA_LEDA','D25_LEDA', 'BA_LEDA', 'Z_LEDA',
                   'SB_D25_LEDA', 'FLUX_G', 'FLUX_R','FLUX_Z', 'FLUX_W1', 
                   'FLUX_W2', 'FLUX_W3', 'FLUX_W4', 'NUVmag'])
df = file.to_pandas()

df = df[df.Z_LEDA > 0]
print(df.Z_LEDA)