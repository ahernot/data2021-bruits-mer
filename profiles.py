import numpy as np
from scipy import interpolate

# TEMPERATURE PROFILE
temperature_profile = np.array([
    [0, 22],
    [-100, 21],
    [-200, 20],
    [-300, 16],
    [-400, 10],
    [-500, 4],
    [-600, 3.5],
    [-700, 3],
    [-800, 3],
    [-900, 3],
    [-1000, 3],
    [-1100, 3],
    [-1200, 3],
    [-1300, 3],
    [-1400, 3],
    [-1500, 3],
    [-10000, 3]
])
calc_T = interpolate.interp1d(temperature_profile.T[0][::-1], temperature_profile.T[1][::-1], kind='quadratic')


# SALINITY PROFILE
S = 35
calc_S = lambda z: np.ones_like(z) * S


# PH PROFILE
pH_profile = np.array([
    [0, 7.98],
    [-100, 7.88],
    [-200, 7.84],
    [-250, 7.87],
    [-300, 7.85],
    [-400, 7.79],
    [-450, 7.7],
    [-500, 7.7],
    [-600, 7.71],
    [-800, 7.73],
    [-1000, 7.72],
    [-2000, 7.73],
    [-3000, 7.74],
    [-10000, 7.74]
])
calc_pH = interpolate.interp1d(pH_profile.T[0][::-1], pH_profile.T[1][::-1], kind='linear')


# Calculate sound velocity gradient
calc_calc_dz_c = lambda calc_c, z_min, z_max, samples: interpolate.interp1d(np.linspace(z_min, z_max, samples), np.gradient(calc_c(np.linspace(z_min, z_max, samples))), kind='quadratic')

