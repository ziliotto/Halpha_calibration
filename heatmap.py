import numpy as np
from scipy.optimize import curve_fit
from multiprocessing import Pool


def blackbody(wavelength, T):
    """
    Planck's law, which describes the black body radiation
    of a source in thermal equilibrium at a given temperature T.
    """
    h = 4.136e-15 # eV/s
    kb = 8.617e-5 # eV/K
    c = 2.998e18 # A/s
    B = 2 * h * c**2 / (wavelength**5 * (np.exp(h*c / (wavelength*kb*T)) - 1))
    return B


def fit(i,j):
    """
    Fits g, r and i fluxes from stacked images to the Planck's law.
    """
    fluxes = np.array([g_s[i][j], r_s[i][j], i_s[i][j]])
    if (fluxes != 0).all():
        try:
            popt, pcov = curve_fit(blackbody, wavelength, fluxes, bounds=(0,1e5))
            T = popt
            print(T)
        except RuntimeError:
            T = 3
            print("curve_fit failed for point", i,j)
    else:
        T = 3
        print(T)
    return float(T)


if __name__ == "__main__":
    
    g_s = np.load('/data1/ziliotto/Halpha/conv_g_galaxy33.npy')
    r_s = np.load('/data1/ziliotto/Halpha/conv_r_galaxy33.npy')
    i_s = np.load('/data1/ziliotto/Halpha/conv_i_galaxy33.npy')

    g_w = 4773.99
    r_w = 6444.80
    i_w = 7858.77
    wavelength = np.array([g_w,r_w,i_w])

    x_dim=r_s.shape[0]
    y_dim=r_s.shape[1]

    tmp = [(x,y) for x in range(x_dim) for y in range(y_dim)]

    with Pool(34) as pool:
        result = np.asarray(list(pool.starmap(fit,tmp)))
        heatmap = result.reshape((x_dim,y_dim))
        np.save('/data1/ziliotto/Halpha/calibrate_line/T_conv_galaxy33_2023.npy',heatmap)
