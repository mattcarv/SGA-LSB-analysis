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
df = df[df.FLUX_W1 > 0]
df = df[df.FLUX_W2 > 0]
df = df[df.FLUX_W3 > 0]
df = df[df.FLUX_W4 > 0]
#%%
# Cleaning the sample - getting rid of extremes

mean_fw3 = df['FLUX_W3'].mean()
std_fw3 = df['FLUX_W3'].std()
mean_fw4 = df['FLUX_W4'].mean()
std_fw4 = df['FLUX_W4'].std()

fw3_cutoff = mean_fw3 + 2*std_fw3
fw4_cutoff = mean_fw4 + 2*std_fw4

df = df[df['FLUX_W3'] < fw3_cutoff]
df = df[df['FLUX_W4'] < fw4_cutoff]

#%%

# Calculating luminosities, SFR and Stellar Mass
# Assigning a distance column
d = cosmo.luminosity_distance(df['Z_LEDA'])
dist = np.array(d)
df['DIST'] = dist


def MagCalc (f, d):
    
    d_mpc = d*(10**6)
    
    m = 22.5 - (2.5*np.log10(f))
    abs_m = m - 5*(np.log10(d_mpc)-1)
    
    return abs_m

mag1 = MagCalc(df['FLUX_W1'], df['DIST'])
mag2 = MagCalc(df['FLUX_W2'], df['DIST'])
mag3 = MagCalc(df['FLUX_W3'], df['DIST'])
mag4 = MagCalc(df['FLUX_W4'], df['DIST'])

def LumCalc (mag, filt):
    
    
    L = 10**(-0.4*(mag-filt))
    
    return L

df['LUM_W1'] = LumCalc(mag1, 3.24)
df['LUM_W2'] = LumCalc(mag2, 3.27)
df['LUM_W3'] = LumCalc(mag3, 3.23)
df['LUM_W4'] = LumCalc(mag4, 3.25)

# Scaling W3 and W4 with the W1 light

df['LUM_W3'] = df['LUM_W3']-(0.158*df['LUM_W1'])
df['LUM_W4'] = df['LUM_W4']-(0.059*df['LUM_W1'])
df = df[df.LUM_W3 > 0]
df = df[df.LUM_W4 > 0]

#df.to_csv('second_subdf_SGA.csv')
# Using a Mass-to-Light ratio to get Stellar Mass

stellar_mass = np.log10(df['LUM_W1'] * 0.6)


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


plt.hist(stellar_mass, bins=10)
plt.plot(x_fit, y_fit, 'r-')

plt.xlabel('log Stellar Mass ($M_{\odot}$)')
plt.ylabel('Probability Density')
plt.show()

# Calculating the SFR based on the W3 and W4 bands
def SFRCalcW3 (lum):
    
    SFR = 10**((0.889*np.log10(lum))-7.76)
    
    return SFR

sfr3 = SFRCalcW3(df['LUM_W3'])

def SFRCalcW4 (lum):
    
    SFR = 10**((0.915*np.log10(lum))-8.01)
    
    return SFR

sfr4 = SFRCalcW4(df['LUM_W4'])



x = np.linspace(-5, 5, 100)
y = x

plt.plot(x, y ,'--', c='k')
plt.scatter(np.log10(sfr4), np.log10(sfr3), alpha=0.5)
plt.xlim(-3, 4.5)
plt.ylim(-3, 4.5)
plt.xlabel('log SFR from the WISE4 band')
plt.ylabel('log SFR from the WISE3 band')
plt.show()