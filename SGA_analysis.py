import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from scipy import stats
from scipy.stats import skewnorm
from scipy.optimize import curve_fit
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
plt.rcParams["figure.figsize"] = [10, 8]
plt.rcParams.update({'font.size': 18})

#%% 
# Selection criteria from the original SGA dataset

# Read the data from the FITS file
df = pd.read_csv('C:/Users/mathe/Downloads/subsec_SGA.csv')
df = df[df.Z_LEDA < 0.1]
# df = df[['SGA_ID_1', 'GALAXY', 'RA_LEDA', 'DEC_LEDA', 'MORPHTYPE', 'PA_LEDA',
#          'D25_LEDA', 'BA_LEDA', 'Z_LEDA', 'SB_D25_LEDA', 'FLUX_G', 'FLUX_R',
#          'FLUX_Z', 'FLUX_W1', 'FLUX_W2', 'FLUX_W3', 'FLUX_W4']]

# df = df[df.D25_LEDA > 0.25]
# df = df[df.Z_LEDA > 0]
# df = df[df.FLUX_W1 > 0]
# df = df[df.FLUX_W2 > 0]
# df = df[df.FLUX_W3 > 0]
# df = df[df.FLUX_W4 > 0]

# df.to_csv('subsec_SGA.csv')

# Create a histogram
hist, bins = np.histogram(df['SB_D25_LEDA'], bins=1000)

# Get the total number of data points
total_points = len(df['SB_D25_LEDA'])

# Calculate the number of data points in the lower 10% magnitudes
lower_10_percent = int(total_points * 0.9)

# Find the bin edges that correspond to the lower 10% of data
lower_bin_edges = bins[:-
                       1][np.where(np.cumsum(hist) <= lower_10_percent)[0][-1]]

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

low_sb = low_sb.sample(frac=0.5, replace=True, random_state=1)

plt.scatter(low_sb['SB_D25_LEDA'], low_sb['D25_LEDA'], alpha=0.3)
plt.ylabel(
    'Major axis diameter at the $25 \; mag \; arcsec^{-2}$ isophote (arcmin)')
plt.xlabel('Mean Surface Brightness (B band, $mag \; arcsec^{-2}$)')
plt.clf()

#%%
# Calculating luminosities, SFR and Stellar Mass
# Assigning a distance column
d = cosmo.luminosity_distance(low_sb['Z_LEDA'])
dist = np.array(d)
low_sb['DIST'] = dist


def MagCalc (f, d):
    
    d = d*(10**6)
    
    m = 22.5 - (2.5*np.log10(f))
    abs_m = (-5*np.log10(d))+5+m
    
    return abs_m

mag1 = MagCalc(low_sb['FLUX_W1'], low_sb['DIST'])
mag2 = MagCalc(low_sb['FLUX_W2'], low_sb['DIST'])
mag3 = MagCalc(low_sb['FLUX_W3'], low_sb['DIST'])
mag4 = MagCalc(low_sb['FLUX_W4'], low_sb['DIST'])


def LumCalc (mag, filt):
    
    
    L = 10**(-0.4*(mag-filt))
    
    return L

low_sb['LUM_W1'] = LumCalc(mag1, 3.24)
low_sb['LUM_W2'] = LumCalc(mag2, 3.27)
low_sb['LUM_W3'] = LumCalc(mag3, 3.23)
low_sb['LUM_W4'] = LumCalc(mag4, 3.25)



# Scaling W3 and W4 with the W1 light

low_sb['LUM_W3'] = low_sb['LUM_W3']-(0.158*low_sb['LUM_W1'])
low_sb['LUM_W4'] = low_sb['LUM_W4']-(0.059*low_sb['LUM_W1'])
low_sb = low_sb[low_sb.LUM_W3 > 0]
low_sb = low_sb[low_sb.LUM_W4 > 0]
low_sb.to_csv('second_subdf_SGA.csv')
# Using a Mass-to-Light ratio to get Stellar Mass

stellar_mass = np.log10(low_sb['LUM_W1'] * 0.6)

# Plotting the mass distribution of this subsample
hist, bins, patches = plt.hist(stellar_mass, bins=1000, density=True)

def skewed_gaussian(x, a, mean, std, skew):
    
    return a * skewnorm.pdf(x, a=skew, loc=mean, scale=std)

mean_guess = np.mean(stellar_mass)
std_guess = np.std(stellar_mass)
skew_guess = 0

amp_guess = np.max(hist)
initial_guess = [amp_guess, mean_guess, std_guess, skew_guess]

try:
    popt, _ = curve_fit(skewed_gaussian, bins[:-1], hist, p0=initial_guess, gtol=1e-5)
except RuntimeError as e:
    print("Fitting failed:", e)
    popt = initial_guess
    
    
x_fit = np.linspace(min(bins), max(bins), 1000)
y_fit = skewed_gaussian(x_fit, *popt)


plt.hist(stellar_mass, bins=1000, density=True)
plt.plot(x_fit, y_fit, 'r-')

plt.xlabel('log Stellar Mass ($M_{\odot}$)')
plt.ylabel('Probability Density')
plt.clf()

# Calculating the SFR based on the W3 and W4 bands
def SFRCalcW3 (lum):
    
    SFR = 10**((0.889*np.log10(lum))-7.76)
    
    return SFR

sfr3 = SFRCalcW3(low_sb['LUM_W3'])

def SFRCalcW4 (lum):
    
    SFR = 10**((0.915*np.log10(lum))-8.01)
    
    return SFR

sfr4 = SFRCalcW4(low_sb['LUM_W4'])

x = np.linspace(-5, 4, 100)
y = x

plt.plot(x, y ,'--', c='k')
plt.scatter(np.log10(sfr4), np.log10(sfr3), alpha=0.5)
plt.xlim(-3, 4)
plt.ylim(-3, 4)
plt.xlabel('log SFR from the WISE4 band')
plt.ylabel('log SFR from the WISE3 band')
plt.clf()

#%%
# Testing a linear regression to the data
linregress = stats.linregress(np.log10(sfr4), np.log10(sfr3))
regression_line = linregress.slope * np.log10(sfr4) + linregress.intercept
residuals = np.log10(sfr3) - (linregress.slope * np.log10(sfr4) + linregress.intercept)


fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

ax1.plot(x, y ,'--', c='k', label='One to one fit')
ax1.scatter(np.log10(sfr4), np.log10(sfr3), alpha=0.5)
ax1.plot(np.log10(sfr4), regression_line, linestyle='dashdot', c='r', 
         label=f'Linear Regression (y = {linregress.slope:.4f}x + {linregress.intercept:.4f})'
         , alpha=0.8)
plt.xlim(-3, 3.2)
plt.ylim(-3, 3.2)
ax1.legend()
ax1.set_ylabel('SFR from the WISE3 band')

ax2.scatter(np.log10(sfr4), residuals, color='green', alpha=0.5)
ax2.axhline(y=0, color='k', linestyle='--')
ax2.set_xlabel('SFR from the WISE4 band')
ax2.set_ylabel('Residuals')
plt.show()

# Statistical tests for the correlation between the two SFR methods
r, p_value = stats.pearsonr(np.log10(sfr4), np.log10(sfr3))
r_squared = linregress.rvalue**2


print(f"Pearson's correlation coefficient (r): {r}")
print(f"Fitted Regression Line (y = {linregress.slope:.2f}x + {linregress.intercept:.2f})")
print(f"R-squared for Fitted Regression Line: {r_squared:.2f}")
#%%
# Plotting a Star Formation Main Sequence with each relation

def curve_function(x, a, b, c):
    return a * x - b * x**2 + c * x**3

x = stellar_mass
y = np.log10(sfr4)

params, _ = curve_fit(curve_function, stellar_mass, np.log10(sfr4))

x_fit = np.linspace(min(x), 11.5, 100)
y_fit = curve_function(x_fit, *params)

y_dotted = curve_function(x_fit, params[0], params[1], params[2])

plt.scatter(x, y, c=low_sb.Z_LEDA, cmap='cool', alpha=0.5)
plt.plot(x_fit, y_fit, 'k', linewidth=2, label='SGA 10% dimmest SFMS')
plt.plot(x_fit, y_dotted+0.4, 'k-.', linewidth=1)
plt.plot(x_fit, y_dotted-0.4, 'k-.', linewidth=1)
plt.xlabel('log Stellar Mass ($M_{\odot}$)')
plt.ylabel('log SFR from the WISE4 band ($M_{\odot} \; yr^{-1}$)')
plt.xlim(6, 11.5)
plt.ylim(-4, 2.8)

cbar = plt.colorbar()
cbar.set_label('Redshift')

# Parameters from the XCOLDGASS SFMS
a_xc = -4.460746
b_xc = -0.836844
c_xc = -0.039050

x_new = np.linspace(min(x), 11.5, 100)
y_new = curve_function(x_new, a_xc, b_xc, c_xc)

plt.plot(x_new, y_new, 'r', linewidth=2, label='XCOLDGASS SFMS')
plt.plot(x_new, y_new+0.4, 'r-.', linewidth=1)
plt.plot(x_new, y_new-0.4, 'r-.', linewidth=1)

plt.legend()
plt.show()

print("Fitted Parameters:")
print("a:", params[0])
print("b:", params[1])
print("c:", params[2])
#%%
# Using the Specific Star Formation (sSFR)

def curve_function(x, a, b, c):
    return a * x - b * x**2 + c * x**3

x = stellar_mass
y = np.log10(sfr3)-stellar_mass

params, _ = curve_fit(curve_function, x, y)

x_fit = np.linspace(min(x), 11.5, 100)
y_fit = curve_function(x_fit, *params)

y_dotted = curve_function(x_fit, params[0], params[1], params[2])

plt.scatter(x, y, c=low_sb.Z_LEDA, cmap='cool', alpha=0.5)
plt.plot(x_fit, y_fit, 'k', linewidth=2, label='SGA 10% dimmest SFMS')
plt.plot(x_fit, y_dotted+0.4, 'k-.', linewidth=1)
plt.plot(x_fit, y_dotted-0.4, 'k-.', linewidth=1)
plt.xlabel('log Stellar Mass ($M_{\odot}$)')
plt.ylabel('log sSFR from the WISE3 band ($yr^{-1}$)')
plt.xlim(6, 11.5)
# plt.ylim(-4, 2.8)

cbar = plt.colorbar()
cbar.set_label('Redshift')

plt.legend()
plt.show()

print("Fitted Parameters:")
print("a:", params[0])
print("b:", params[1])
print("c:", params[2])
#%%
# Testing the conversion from Lee et al. 2013

def SFRCalcW3 (lum):
    
    SFR = 9.54 * (10e-10) * (lum**1.03)
    
    return SFR

sfr3_lee = SFRCalcW3(low_sb['LUM_W3'])

def SFRCalcW4 (lum):
    
    SFR = 4.25 * (10e-9) * (lum**0.96)
    
    return SFR

sfr4_lee = SFRCalcW4(low_sb['LUM_W4'])

x = np.linspace(-5, 4.5, 100)
y = x

lin_reg = stats.linregress(np.log10(sfr3_lee), np.log10(sfr4_lee))
reg_line = lin_reg.slope * np.log10(sfr3_lee) + lin_reg.intercept

r_coef, p_value_coef = stats.pearsonr(np.log10(sfr3_lee), np.log10(sfr4_lee))
r_squared_lee = lin_reg.rvalue**2

plt.scatter(np.log10(sfr3_lee), np.log10(sfr4_lee), alpha=0.5)
plt.plot(x, y ,'--', c='k')
plt.plot(np.log10(sfr3_lee), reg_line, linestyle='dashdot', c='r', 
         label=f'Linear Regression (y = {lin_reg.slope:.4f}x + {lin_reg.intercept:.4f})'
         , alpha=0.8)
plt.xlim(-3, 4)
plt.ylim(-3, 4)
plt.xlabel('log SFR from the WISE3 band')
plt.ylabel('log SFR from the WISE4 band')


print(f"Pearson's correlation coefficient (r): {r_coef}")
print(f"Fitted Regression Line (y = {lin_reg.slope:.2f}x + {lin_reg.intercept:.2f})")
print(f"R-squared for Fitted Regression Line: {r_squared_lee:.2f}")


plt.legend()
plt.show()
#%%

# SFMS from the Lee scaling relation

def curve_function(x, a, b, c):
    return a * x - b * x**2 + c * x**3

x_lee = stellar_mass
y_lee = np.log10(sfr3_lee)

params, _ = curve_fit(curve_function, x_lee, y_lee)

x_fit = np.linspace(min(x), 11.7, 100)
y_fit = curve_function(x_fit, *params)

plt.scatter(x_lee, y_lee, c=low_sb.Z_LEDA, cmap='viridis', alpha=0.5)
plt.plot(x_fit, y_fit, c='k', linewidth=2)
plt.xlim(6, 11.6)
plt.ylim(-3, 4.5)
plt.xlabel('log SFR from the WISE3 band')
plt.ylabel('log SFR from the WISE4 band')

x_new = np.linspace(min(x), 11.7, 100)
y_new = curve_function(x_new, a_xc, b_xc, c_xc)

plt.plot(x_new, y_new, 'r', linewidth=2, label='XCOLDGASS SFMS')
plt.plot(x_new, y_new+0.4, 'r-.', linewidth=1)
plt.plot(x_new, y_new-0.4, 'r-.', linewidth=1)

cbar = plt.colorbar()
cbar.set_label('Redshift')

plt.clf()