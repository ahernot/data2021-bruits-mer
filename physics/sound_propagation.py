import numpy as np

# check this out https://monrecifamoi.saulme.fr/salinite/sea_water_calculator_teos10.php

def calc_rho (S, T, p):
    """
    IES 80 – High Pressure International Equation of State of Seawater
    https://unesdoc.unesco.org/ark:/48223/pf0000047363

    :param S: Practical salinity (in ‰), range:   0‰ ≤ S ≤ 42‰
    :param T: Temperature (in ℃),        range:  -2℃ ≤ T ≤ 40℃
    :param p: Applied pressure (in bar), range: 0bar ≤ p ≤ 1000bar
    :return: Seawater density (in kg.m^-3)
    """

    # Density of the Standard Mean Ocean Water (SMOW) [Bigg, 1967]
    rho_W = 999.842594 + 6.793952e-2 * T - 9.09529e-3 * np.power(T, 2) + 1.001685e-4 * np.power(T, 3) - 1.120083e-6 * np.power(T, 4) + 6.536336e-9 * np.power(T, 5)
    
    # Coefficients
    A = 8.24493e-1 - 4.0899e-3 * T + 7.6438e-5 * np.power(T, 2) - 8.2467e-7 * np.power(T, 3) + 5.3875e-9 * np.power(T, 4)
    B = -5.72466e-3 + 1.0227e-4 * T - 1.6546e-6 * np.power(T, 2)
    C = 4.8314e-4

    # One Atmosphere International Equation of State of Seawater (1980) (standard error = 3.6e-3 kg.m-3; validity 0‰≤S≤42‰, -2℃≤T≤40℃)
    rho_1atm = rho_W + A * S + B * np.power(S, 1.5) + C * np.power(S, 2)



    # Pure water terms
    K_W = 19652.21 + 148.4206 * T - 2.327105 * np.power(T, 2) + 1.360477e-2 * np.power(T, 3) - 5.155288e-5 * np.power(T, 4)
    A_W = 3.239908 + 1.43713e-3 * T + 1.16092e-4 * np.power(T, 2) - 5.79905e-7 * np.power(T, 3)
    B_W = 8.50935e-5 - 6.12293e-6 * T + 5.2787e-8 * np.power(T, 2)

    K_1atm = K_W + \
        (54.6746 - 0.603459 * T + 1.09987e-2 * np.power(T, 2) - 6.1670e-5 * np.power(T, 3)) * S + \
        (7.944e-2 + 1.6483e-2 * T - 5.3009e-4 * np.power(T, 2)) * np.power(S, 1.5)
    A = A_W + \
        (2.2838e-3 - 1.0981e-5 * T - 1.6078e-6 * np.power(T, 2)) * S + \
        1.91075e-3 * np.power(S, 1.5)
    B = B_W + \
        (-9.9348e-7 + 2.0816e-8 * T + 9.1697e-10 * np.power(T, 2)) * S

    # Secant Bulk Modulus
    K = K_1atm + A * p + B * np.power(p, 2)

    # High Pressure International Equation of State of Seawater
    rho = rho_1atm / (1 - p / K)
    

    return rho


def calc_dS_rho (S, T, p):
    """
    IES 80 – High Pressure International Equation of State of Seawater
    https://unesdoc.unesco.org/ark:/48223/pf0000047363

    :param S: Practical salinity (in ‰), range:   0‰ ≤ S ≤ 42‰
    :param T: Temperature (in ℃),        range:  -2℃ ≤ T ≤ 40℃
    :param p: Applied pressure (in bar), range: 0bar ≤ p ≤ 1000bar
    :return: Seawater density (in kg.m^-3)
    """

    # Density of the Standard Mean Ocean Water (SMOW) [Bigg, 1967]
    rho_W = 999.842594 + 6.793952e-2 * T - 9.09529e-3 * np.power(T, 2) + 1.001685e-4 * np.power(T, 3) - 1.120083e-6 * np.power(T, 4) + 6.536336e-9 * np.power(T, 5)
    
    # Coefficients
    A = 8.24493e-1 - 4.0899e-3 * T + 7.6438e-5 * np.power(T, 2) - 8.2467e-7 * np.power(T, 3) + 5.3875e-9 * np.power(T, 4)
    B = -5.72466e-3 + 1.0227e-4 * T - 1.6546e-6 * np.power(T, 2)
    C = 4.8314e-4

    # One Atmosphere International Equation of State of Seawater (1980) (standard error = 3.6e-3 kg.m-3; validity 0‰≤S≤42‰, -2℃≤T≤40℃)
    rho_1atm = rho_W + A * S + B * np.power(S, 1.5) + C * np.power(S, 2)
    dS_rho_1atm = A + 1.5 * B * np.power(S, 0.5) + 2 * C * S  # derivative



    # Pure water terms
    K_W = 19652.21 + 148.4206 * T - 2.327105 * np.power(T, 2) + 1.360477e-2 * np.power(T, 3) - 5.155288e-5 * np.power(T, 4)
    A_W = 3.239908 + 1.43713e-3 * T + 1.16092e-4 * np.power(T, 2) - 5.79905e-7 * np.power(T, 3)
    B_W = 8.50935e-5 - 6.12293e-6 * T + 5.2787e-8 * np.power(T, 2)

    K_1atm = K_W + \
        (54.6746 - 0.603459 * T + 1.09987e-2 * np.power(T, 2) - 6.1670e-5 * np.power(T, 3)) * S + \
        (7.944e-2 + 1.6483e-2 * T - 5.3009e-4 * np.power(T, 2)) * np.power(S, 1.5)
    dS_K_1atm = (54.6746 - 0.603459 * T + 1.09987e-2 * np.power(T, 2) - 6.1670e-5 * np.power(T, 3)) + \
        1.5 * (7.944e-2 + 1.6483e-2 * T - 5.3009e-4 * np.power(T, 2)) * np.power(S, 0.5)
    A = A_W + \
        (2.2838e-3 - 1.0981e-5 * T - 1.6078e-6 * np.power(T, 2)) * S + \
        1.91075e-3 * np.power(S, 1.5)
    dS_A_1atm = (2.2838e-3 - 1.0981e-5 * T - 1.6078e-6 * np.power(T, 2)) + \
        1.5 * 1.91075e-3 * np.power(S, 0.5)
    B = B_W + \
        (-9.9348e-7 + 2.0816e-8 * T + 9.1697e-10 * np.power(T, 2)) * S
    dS_B_1atm = -9.9348e-7 + 2.0816e-8 * T + 9.1697e-10 * np.power(T, 2)

    # Secant Bulk Modulus
    K = K_1atm + A * p + B * np.power(p, 2)
    dS_K = dS_K_1atm + dS_A_1atm * p + dS_B_1atm * np.power(p, 2)

    # High Pressure International Equation of State of Seawater
    denom = 1 - p / K
    dS_denom = p * dS_K / np.power(K, 2)
    rho = rho_1atm / denom
    dS_rho = (dS_rho_1atm * denom - rho_1atm * dS_denom) / np.power(denom, 2)
    
    return dS_rho


def calc_dp_rho (S, T, p):
    """
    IES 80 – High Pressure International Equation of State of Seawater
    https://unesdoc.unesco.org/ark:/48223/pf0000047363

    :param S: Practical salinity (in ‰), range:   0‰ ≤ S ≤ 42‰
    :param T: Temperature (in ℃),        range:  -2℃ ≤ T ≤ 40℃
    :param p: Applied pressure (in bar), range: 0bar ≤ p ≤ 1000bar
    :return: Seawater density (in kg.m^-3)
    """

    # Density of the Standard Mean Ocean Water (SMOW) [Bigg, 1967]
    rho_W = 999.842594 + 6.793952e-2 * T - 9.09529e-3 * np.power(T, 2) + 1.001685e-4 * np.power(T, 3) - 1.120083e-6 * np.power(T, 4) + 6.536336e-9 * np.power(T, 5)
    
    # Coefficients
    A = 8.24493e-1 - 4.0899e-3 * T + 7.6438e-5 * np.power(T, 2) - 8.2467e-7 * np.power(T, 3) + 5.3875e-9 * np.power(T, 4)
    B = -5.72466e-3 + 1.0227e-4 * T - 1.6546e-6 * np.power(T, 2)
    C = 4.8314e-4

    # One Atmosphere International Equation of State of Seawater (1980) (standard error = 3.6e-3 kg.m-3; validity 0‰≤S≤42‰, -2℃≤T≤40℃)
    rho_1atm = rho_W + A * S + B * np.power(S, 1.5) + C * np.power(S, 2)



    # Pure water terms
    K_W = 19652.21 + 148.4206 * T - 2.327105 * np.power(T, 2) + 1.360477e-2 * np.power(T, 3) - 5.155288e-5 * np.power(T, 4)
    A_W = 3.239908 + 1.43713e-3 * T + 1.16092e-4 * np.power(T, 2) - 5.79905e-7 * np.power(T, 3)
    B_W = 8.50935e-5 - 6.12293e-6 * T + 5.2787e-8 * np.power(T, 2)

    K_1atm = K_W + \
        (54.6746 - 0.603459 * T + 1.09987e-2 * np.power(T, 2) - 6.1670e-5 * np.power(T, 3)) * S + \
        (7.944e-2 + 1.6483e-2 * T - 5.3009e-4 * np.power(T, 2)) * np.power(S, 1.5)
    A = A_W + \
        (2.2838e-3 - 1.0981e-5 * T - 1.6078e-6 * np.power(T, 2)) * S + \
        1.91075e-3 * np.power(S, 1.5)
    B = B_W + \
        (-9.9348e-7 + 2.0816e-8 * T + 9.1697e-10 * np.power(T, 2)) * S

    # Secant Bulk Modulus
    K = K_1atm + A * p + B * np.power(p, 2)
    dp_K = A + 2 * B * p

    # High Pressure International Equation of State of Seawater
    denom = 1 - p / K
    dp_denom = -1 / K + p / np.power(K, 2) * dp_K
    rho = rho_1atm / (1 - p / K)
    dp_rho = -1 * rho_1atm * dp_denom / np.power(denom, 2)
    

    return dp_rho


def freezing_temperature (S, p):
    """
    Millero & Leung equation (1976)
    https://cahier-de-prepa.fr/tsi2-riquet/download?id=705

    :param S: Practical salinity (in ‰)
    :param p: Applied pressure (in bar)
    :return: Freezing temperature (in ℃)
    """
    T_f = -0.0575 * S + 1.710523e-3 * np.power(S, 1.5) - 2.154996e-4 * np.power(S, 2) - 7.53e-4 * p
    return T_f





### COEFFICIENTS ÉLASTIQUES
def calc_alpha_V (S, T, p):
    """
    Cubic thermal expansion coefficient (alpha_V) – Coefficient de dilatation thermique
    """
    rho = calc_rho(S, T, p)
    raise NotImplementedError
    dT_rho = calc_dT_rho(S, T, p)
    return -1 * dT_rho / rho

def calc_beta (S, T, p):
    """
    Salinity contraction coefficient (beta) – Coefficient de contraction haline
    """
    rho = calc_rho(S, T, p)
    dS_rho = calc_dS_rho(S, T, p)
    return dS_rho / rho

def calc_kappa_T (S, T, p):
    """
    Compressibility (kappa_T) – Coefficient de compressibilité isotherme
    """
    rho = calc_rho(S, T, p)
    dp_rho = calc_dp_rho(S, T, p)
    return dp_rho / rho

def sound_velocity (S, T, p):
    raise NotImplementedError
    kappa_T = calc_kappa_T(S, T, p)
    rho = calc_rho(S, T, p)
    return np.power(1 / (kappa_T * rho), 0.5)



def sound_velocity_simplified (S, T, z):
    """
    Sound velocity in seawater (simplified formula)
    http://lecalve.univ-tln.fr/oceano/fiches/fiche3F.htm

    :param S: Practical salinity (in ‰)
    :param T: Temperature (in ℃)
    :param z: Altitude (in m)
    :return: Sound velocity (in m.s^-1)
    """

    c = 1410 + 4.21 * T - 0.037 * np.power(T, 2) + 1.1 * S - 0.018 * z
    
    return c


def sound_velocity_leroy (S, T, z):
    """
    Sound velocity in seawater (Claude Leroy)
    http://lecalve.univ-tln.fr/oceano/fiches/fiche3F.htm

    :param S: Practical salinity (in ‰), range:     30‰ ≤ S ≤ 42‰
    :param T: Temperature (in ℃),        range:     -2℃ ≤ T ≤ 24.5℃
    :param z: Altitude (in m),           range: -1000m ≤ z ≤ 0m
    :return: Sound velocity (in m.s^-1)
    """

    c = 1492.9 + \
        3 * (T-10) - 0.006 * np.power(T-10, 2) - 0.04 * np.power(T-18, 2) + \
        1.2 * (S-35) - 0.01 * (T-18) * (S-35) - \
        z / 61
    
    return c
    

def sound_velocity_medwin (S, T, z):
    """
    Sound velocity in seawater (H. Medwin)
    http://lecalve.univ-tln.fr/oceano/fiches/fiche3F.htm

    :param S: Practical salinity (in ‰), range:     0‰ ≤ S ≤ 45‰
    :param T: Temperature (in ℃),        range:     0℃ ≤ T ≤ 35℃
    :param z: Altitude (in m),           range: -1000m ≤ z ≤ 0m
    :return: Sound velocity (in m.s^-1)
    """

    c = 1449.2 + \
        4.6 * T - 0.055 * np.power(T, 2) + 0.00029 * np.power(T, 3) + \
        (1.34 - 0.01 * T) * (S - 35) - \
        0.016 * z 
    
    return c


def sound_velocity_mackenzie (S, T, z):
    """
    Sound velocity in seawater (Mackenzie)
    http://lecalve.univ-tln.fr/oceano/fiches/fiche3F.htm

    :param S: Practical salinity (in ‰), range:     0‰ ≤ S ≤ 40‰
    :param T: Temperature (in ℃),        range:     0℃ ≤ T ≤ 30℃
    :param z: Altitude (in m),           range: -1000m ≤ z ≤ 0m
    :return: Sound velocity (in m.s^-1)
    """

    c = 1448.96 + \
        4.591 * T - 0.05304 * np.power(T, 2) + 0.0002374 * np.power(T, 3) + \
        1.34 * (S-35) - 0.01025 * T * (S-35) - \
        0.016 * z + 0.0000001675 * np.power(z, 2)
    
    return c








