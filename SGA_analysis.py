import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table

# Read the data from the FITS file
file = Table.read('C:/Users/mathe/Downloads/SGA-2020.fits')
df = file.to_pandas()
df = df[df.D25_LEDA>0.25]

# Create a histogram
hist, bins = np.histogram(df['SB_D25_LEDA'], bins=1000)

# Calculate the total number of data points
total_points = len(df['SB_D25_LEDA'])

# Calculate the number of data points in the lower 10%
lower_10_percent = int(total_points * 0.9)

# Find the bin edges that correspond to the lower 10% of data
lower_bin_edges = bins[:-1][np.where(np.cumsum(hist) <= lower_10_percent)[0][-1]]

# Select data points in the lower 10%
low_sb = df[df['SB_D25_LEDA'] >= lower_bin_edges]

plt.hist(df['SB_D25_LEDA'], bins=1000)
plt.axvline(x=lower_bin_edges, color='red', linestyle='--', label='10% Cutoff')
plt.ylabel('Number')
plt.xlabel('Mean Surface Brightness (B band, $mag \; arcsec^{-2}$)')
plt.legend()
plt.clf()

# Calculate the mean and standard deviation
mean_sb = low_sb['SB_D25_LEDA'].mean()
std_sb = low_sb['SB_D25_LEDA'].std()
mean_d = low_sb['D25_LEDA'].mean()
std_d = low_sb['D25_LEDA'].std()

sb_cutoff = mean_sb + 2*std_sb
d_cutoff = mean_d + 2*std_d

low_sb = low_sb[low_sb['SB_D25_LEDA'] < sb_cutoff]
low_sb = low_sb[low_sb['D25_LEDA'] < d_cutoff]


plt.scatter(low_sb['SB_D25_LEDA'], low_sb['D25_LEDA'])
plt.show()