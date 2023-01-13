import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as interp
import os.path
from scipy import integrate
from multiprocessing import Pool


def readfile(filename):
    ''' Reads a file.
    Input: Filename.
    Returns: x, y, xmin, xmax, number of points, filename.
    '''
    if os.path.isfile(filename) != True:
        print("File not in this directory!")
    else:
        file = np.loadtxt(filename, unpack=True)
        x = file[0]
        y = file[1]
        xmax = np.amax(x)
        xmin = np.amin(x)
        npts = len(file[0])
        return(x, y, xmin, xmax, npts, filename)

    
def continuum(T):
    """
    Planck's law, which describes the black body radiation
    of a source in thermal equilibrium at a given temperature T.
    (Atomic units, distance units in Angstroms)
    """
    h = 6.6261e-27     # cm2 g s-1
    kb = 1.3807e-16    # cm2 g s-2 K-1
    c = 2.99792458e10  # cm/s
    B = lambda x: 2 * h * c**2 / (x**5 * (np.exp(h*c / (x*kb*T)) - 1)) # 1/A**3
    return B


def obsWavelength(x, vel):
    ''' Corrects for systemic velocity.
    Input: Rest wavelength and velocity in cm/s.
    Output: Observed wavelengh.
    '''
    c = 2.99792458e10 # cm/s
    z = vel / c
    lambda_obs = x * (1 + z)
    return lambda_obs


def a(aux_continuum):   # Fc, phiL
    funcA = lambda x: aux_continuum(x) * lineInterp(x)
    A = integrate.quad( funcA, xlmin, xlmax )
    return A


def p(standInterp, lineInterp):
    funcP = lambda x: standInterp(x) * lineInterp(x) # phi_l
    P = (1 / fsl) * 10 ** ( -0.4 * kpl * Xsl  ) * integrate.quad( funcP, xlmin, xlmax )[0]
    return P


def r_g(fwhm, vsys):
    g = fwhm / (2 * np.log(2))
    cte = 1. / ( np.sqrt( np.pi ) * g )
    waveobs = obsWavelength(6562.81 * 1e-8, vsys)
    funcR = lambda x: lineInterp(x) * np.exp(-(( x - waveobs )/g )**2)
    intR = integrate.quad(funcR, xlmin, xlmax)
    R = intR[0] * cte
    return(R)


def mod_q(aux_continuum):
    funcQ1g = lambda x: standInterp(x) * contgInterp(x) # phi_c (contInterp)
    funcQ2g = lambda x: aux_continuum(x) * contgInterp(x) # phi_c
    funcQ1r = lambda x: standInterp(x) * contrInterp(x) # phi_c (contInterp)
    funcQ2r = lambda x: aux_continuum(x) * contrInterp(x) # phi_c
    funcQ1i = lambda x: standInterp(x) * contiInterp(x) # phi_c (contInterp)
    funcQ2i = lambda x: aux_continuum(x) * contiInterp(x) # phi_c
    constant = (10**( 0.4 * (kpc_g + kpc_r + kpc_i) * (-(Xsc_g + Xsc_r + Xsc_i)) )) / (fsc_g + fsc_r + fsc_i)
    integrals = ( integrate.quad(funcQ1g, xgmin, xgmax, epsabs=1.49e-11)[0] / integrate.quad(funcQ2g, xgmin, xgmax, epsabs=1.49e-11)[0] ) + ( integrate.quad(funcQ1r, xrmin, xrmax, epsabs=1.49e-11)[0] / integrate.quad(funcQ2r, xrmin, xrmax, epsabs=1.49e-11)[0] ) + ( integrate.quad(funcQ1i, ximin, ximax, epsabs=1.49e-11)[0] / integrate.quad(funcQ2i, ximin, ximax, epsabs=1.49e-11)[0] )
    Q = constant * integrals
    return Q


def halpha(i,j):
    T = T_s[i][j]
    if T <=40: # To avoid errors in the integrals because of too small temperatures
        Fl = 0
    else:
        aux_continuum = continuum(T)
        Fl = ( 1 / R ) * (( n662_s[i][j] / ((g_s[i][j] + r_s[i][j] + i_s[i][j])) ) * (P/((mod_q(aux_continuum)))) -a(aux_continuum)[0] )
    return Fl


if __name__ == "__main__":
    
    # flux and temperature data
    n662 = np.load('/data1/ziliotto/Halpha/calibrate_line/test/n662_data.npy')
    T_s = np.load('/data1/ziliotto/Halpha/calibrate_line/T_conv_galaxy33.npy')
    n662_s = n662[11160:11280,17930:18080]
    g_s = np.load('/data1/ziliotto/Halpha/conv_g_galaxy33.npy')
    r_s = np.load('/data1/ziliotto/Halpha/conv_r_galaxy33.npy')
    i_s = np.load('/data1/ziliotto/Halpha/conv_i_galaxy33.npy')

    x_dim=n662_s.shape[0]
    y_dim=n662_s.shape[1]

    tmp = [(x,y) for x in range(x_dim) for y in range(y_dim)]

    # standard data
    cal_u = 10.675 # calibrated magnitudes
    cal_g = 9.443
    cal_r = 9.362
    cal_i = 9.405
    cal_z = 9.460
    mag = np.array([cal_u,cal_g,cal_r,cal_i,cal_z])

    # Wavelengths of each band
    u_w = 3580.78 * 1e-8 # cm
    g_w = 4773.99 * 1e-8
    r_w = 6444.80 * 1e-8
    i_w = 7858.77 * 1e-8
    z_w = 9281.68 * 1e-8
    wavelength = np.array([u_w,g_w,r_w,i_w,z_w])

    u_flux = 4.706e-12 * 1e7 # erg s-1 cm-3 (observed fluxes of the standards)
    print(u_flux)
    g_flux = 7.859e-12 * 1e7
    print(g_flux)
    r_flux = 5.047e-12 * 1e7
    print(r_flux)
    i_flux = 3.226e-12 * 1e7
    print(i_flux)
    z_flux = 2.141e-12 * 1e7
    print(z_flux)
    fluxes_converted = np.array([u_flux,g_flux,r_flux,i_flux,z_flux])

    standx = wavelength
    standy = fluxes_converted
    standInterp = interp(standx,standy)
    xmin = np.amin(standx)
    xmax = np.amax(standx)

    # Line filter
    Xl = 0
    print('Line filter air masses (program objects):', Xl)
    Xsl = 1.4 # same as r band
    print('Line filter air masses (standard objects):', Xsl)

    # Continuum filters
    Xc_g = 0
    print('Air masses for the continuum filter g (program):', Xc_g)
    Xc_r = 0
    print('Air masses for the continuum filter r (program):', Xc_r)
    Xc_i = 0
    print('Air masses for the continuum filter i (program):', Xc_i)

    Xsc_g = 1.4
    print('Air masses for the continuum filter g (standard):', Xsc_g)
    Xsc_r = 1.4
    print('Air masses for the continuum filter r (standard):', Xsc_r)
    Xsc_i = 1.4
    print('Air masses for the continuum filter g (standard):', Xsc_i)

    kpl = 0.02965678919729933
    print('mag/air mass for line:', kpl)

    kpc_g = 0.1939
    print('mag/air mass for continuum g:', kpc_g)
    kpc_r = 0.095
    print('mag/air mass for continuum r:', kpc_r)
    kpc_i = 0.0681
    print('mag/air mass for continuum i:', kpc_i)

    fsc_g = (4328678-5.57509)/3    
    print('Raw fluxes of the standard with sky subtracted (continuum g):', fsc_g)
    fsc_r = (3202252-15.27314)/3    
    print('Raw fluxes of the standard with sky subtracted (continuum r):', fsc_r)
    fsc_i = (1749529-33.96471)/3    
    print('Raw fluxes of the standard with sky subtracted (continuum i):', fsc_i)

    fsl = fsc_r / 7.754
    print('Raw fluxes of the standard with sky subtracted (line):', fsl)

    nline = 1
    print('Number of lines found in the line filter range, its rest wavelengths and fractional contribution (sum=1.0):', nline)

    rfWave = [6562.81 * 1e-8,1]
    print('Rest wavelength of each line as well as its fractional contibution (sum = 1.0):', rfWave)

    vsys = 1700*100000 # cm/s
    print('Systemic velocity of the galaxy:', vsys)

    texpl = 9000
    print('Exposure times of program frames (line):', texpl)

    texpc_g = 6300
    print('Exposure times of program frames (continuum g):', texpc_g)
    texpc_r = 1200
    print('Exposure times of program frames (continuum r):', texpc_r)
    texpc_i = 6600
    print('Exposure times of program frames (continuum i):', texpc_i)
    texpc = texpc_g + texpc_r + texpc_i

    skyl = 0 # sky background is 0 in the stacks
    print('Sky background of the program frames in counts/pixel (line):', skyl)
    skyc_g = 0
    print('Sky background of the program frames in counts/pixel (continuum g):', skyc_g)
    skyc_r = 0
    print('Sky background of the program frames in counts/pixel (continuum r):', skyc_r)
    skyc_i = 0
    print('Sky background of the program frames in counts/pixel (continuum i):', skyc_i)

    # N662
    path = '/data1/ziliotto/Halpha/calibrate_line/'
    line = readfile(os.path.join(path, 'n662.dat'))
    print('Line filter file:', line[5])
    linex = line[0] * 1e-8
    liney = line[1]
    xlmax = line[3] * 1e-8
    xlmin = line[2] * 1e-8

    # g filter
    g_transmission = readfile(os.path.join(path, 'CTIO_DECam.g_filter.dat'))
    print('g filter file:', g_transmission[5])
    xgmax = g_transmission[3]* 1e-8
    xgmin = g_transmission[2]* 1e-8
    contxg = g_transmission[0]* 1e-8
    contyg = g_transmission[1]

    # r filter
    r_transmission = readfile(os.path.join(path, 'CTIO_DECam.r_filter.dat'))
    print('r filter file:', r_transmission[5])
    xrmax = r_transmission[3] * 1e-8
    xrmin = r_transmission[2] * 1e-8
    contxr = r_transmission[0] * 1e-8
    contyr = r_transmission[1]

    # i filter
    i_transmission = readfile(os.path.join(path, 'CTIO_DECam.i_filter.dat'))
    print('i filter file:', i_transmission[5])
    ximax = i_transmission[3]* 1e-8
    ximin = i_transmission[2] * 1e-8
    contxi = i_transmission[0] * 1e-8
    contyi = i_transmission[1]

    # Interpolations: transmission functions
    lineInterp = interp(linex,liney)
    contgInterp = interp(contxg,contyg)
    contrInterp = interp(contxr,contyr)
    contiInterp = interp(contxi,contyi)

    P = p(standInterp, lineInterp)
    R = r_g(15 * 1e-8,vsys)
    
    convert = 1/6.06336e25 # conversion factor for the resulting H-Alpha flux to be in units of erg/s/cm^2/A

    with Pool(32) as pool:
        result = np.asarray(list(pool.starmap(halpha,tmp)))
        Halpha = result.reshape((x_dim,y_dim)) * convert
        np.save('/data1/ziliotto/Halpha/calibrate_line/conv_data/halpha_conv_galaxy33_2023.npy',Halpha)
