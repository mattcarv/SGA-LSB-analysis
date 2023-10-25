import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table

# Read the data from the FITS file
file = Table.read('/home/mdocarm/Downloads/SGA-2020.fits')
df = file.to_pandas()

# Create a histogram
hist, bins = np.histogram(df['SB_D25_LEDA'], bins=1000)

# Calculate the total number of data points
total_points = len(df['SB_D25_LEDA'])

# Calculate the number of data points in the lower 10%
lower_10_percent = int(total_points * 0.1)

# Find the bin edges that correspond to the lower 10% of data
lower_bin_edges = bins[:-1][np.where(np.cumsum(hist) <= lower_10_percent)[0][-1]]

# Select data points in the lower 10%
selected_data = df[df['SB_D25_LEDA'] <= lower_bin_edges]

plt.hist(df['SB_D25_LEDA'], bins=1000)
plt.axvline(x=lower_bin_edges, color='red', linestyle='--', label='10% Cutoff')

plt.legend()
plt.show()

print(selected_data['SB_D25_LEDA'].max())