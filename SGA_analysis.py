import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# Read the data from the FITS file
df = pd.read_csv('/home/mdocarm/Downloads/SGA_xmatch.csv')
df = df[['SGA_ID_1', 'GALAXY', 'RA_LEDA', 'DEC_LEDA', 'MORPHTYPE', 'PA_LEDA',
         'D25_LEDA', 'BA_LEDA', 'Z_LEDA', 'SB_D25_LEDA', 'FLUX_G', 'FLUX_R',
         'FLUX_Z', 'FLUX_W1', 'FLUX_W2', 'FLUX_W3', 'FLUX_W4']]

df = df[df.D25_LEDA>0.25]

# Create a histogram
hist, bins = np.histogram(df['SB_D25_LEDA'], bins=1000)

# Get the total number of data points
total_points = len(df['SB_D25_LEDA'])

# Calculate the number of data points in the lower 10% magnitudes
lower_10_percent = int(total_points * 0.9)

# Find the bin edges that correspond to the lower 10% of data
lower_bin_edges = bins[:-1][np.where(np.cumsum(hist) <= lower_10_percent)[0][-1]]

# Select data points in the lower 10% magnitudes
low_sb = df[df['SB_D25_LEDA'] >= lower_bin_edges]

plt.hist(df['SB_D25_LEDA'], bins=1000)
plt.axvline(x=lower_bin_edges, color='red', linestyle='--', label='10% Cutoff')
plt.ylabel('Number')
plt.xlabel('Mean Surface Brightness (B band, $mag \; arcsec^{-2}$)')
plt.legend()
plt.clf()

# Calculate the mean and standard deviation for each
mean_sb = low_sb['SB_D25_LEDA'].mean()
std_sb = low_sb['SB_D25_LEDA'].std()
mean_d = low_sb['D25_LEDA'].mean()
std_d = low_sb['D25_LEDA'].std()

sb_cutoff = mean_sb + 2*std_sb
d_cutoff = mean_d + 2*std_d

low_sb = low_sb[low_sb['SB_D25_LEDA'] < sb_cutoff]
low_sb = low_sb[low_sb['D25_LEDA'] < d_cutoff]

low_sb = low_sb.sample(frac=0.7, replace=True, random_state=1)

plt.scatter(low_sb['SB_D25_LEDA'], low_sb['D25_LEDA'], alpha=0.3)
plt.ylabel('Major axis diameter at the $25 \; mag \; arcsec^{-2}$ isophote (arcmin)')
plt.xlabel('Mean Surface Brightness (B band, $mag \; arcsec^{-2}$)')
plt.clf()

# General correlation plot to understand the possible trends in the dataset
# import seaborn as sns

# corr = low_sb.corr()
# mask = np.triu(np.ones_like(corr, dtype=bool))
# sns.heatmap(corr, mask=mask, cmap='RdBu', center=0,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Calculating luminosities

def LumCalc(f, z):
    
    d = cosmo.luminosity_distance(z)*(10^(6))
    f = f > 0
    sol_m = 3.24
    
    m = 22.5 - (2.5*np.log10(f))
    abs_m = -5*np.log10(d)+5+m
    
    l = 10^((-0.4)*(abs_m-sol_m))
    
    return l
    
#lum = LumCalc(low_sb['FLUX_W1'], low_sb['Z_LEDA'])
print(np.log10(cosmo.luminosity_distance(0.073)*(1000000)))
