import numpy as np

import matplotlib.pyplot as plt

def absorption (f: float, depth: float, temperature: float, salinity: float, pH: float):
    """
    Ainslie and McColm 1998
    :param f: Signal frequency (in kHz)
    :param depth: Depth (in km)
    :param temperature: Temperature (in °C)
    :param salinity: Salinity (in ppt)
    :param pH: Acidity (pH)
    """
    # http://resource.npl.co.uk/acoustics/techguides/seaabsorption/
    # http://resource.npl.co.uk/acoustics/techguides/seaabsorption/physics.html


    """
    ! specify range of validity
    0 < distance < 7km
    """

    # Boric acid (relaxation absorption)
    f1 = 0.78 * np.sqrt(salinity / 35) * np.exp(temperature / 26)  # Boric acid relaxation frequency (kHz)
    A1 = 0.106
    alpha_1 = A1 * (f1 * np.power(f, 2)) / (np.power(f1, 2) + np.power(f, 2))

    # Magnesium sulfate (relaxation absorption)
    f2 = 42 * np.exp(temperature / 17)  # Magnesium sulfate relaxation frequency (kHz)
    A2 = 0.52 * (1 + temperature / 43) * (salinity / 35)
    alpha_2 =  A2 * (f2 * np.power(f, 2)) / (np.power(f2, 2) + np.power(f, 2))

    # Pure water (visquous absorption)
    A3 = 0.00049
    alpha_3 = A3 * np.power(f, 2)

    # Total seawater absorption (in dB/km)
    alpha = alpha_1 * np.exp((pH - 8) / 0.56) +\
            alpha_2 * np.exp(-1 * depth / 6)  +\
            alpha_3 * np.exp(-1 * (temperature/27 + depth/17))


    return alpha



# kwargs = {
#     'depth': 0.05,
#     'temperature': 8,
#     'salinity': 35,
#     'pH': 8
# }

# print(absorption(1, **kwargs))

# frequencies = np.linspace (1e-1, 1e3, 100)
# absorptions = absorption (frequencies, **kwargs)

# plt.figure (figsize=(15, 10))
# plt.loglog(frequencies, absorptions)
# plt.xlabel('Frequency (kHz)')
# plt.ylabel('Absorption (dB/km)')
# plt.show()



# absorption_10hz = absorption(10, **kwargs)
# print(absorption_10hz)

# plt.figure(figsize=(15, 10))
# distances = np.linspace(0, 7, 200)  # in km
# plt.plot(distances, absorption_10hz * distances)
# plt.show()