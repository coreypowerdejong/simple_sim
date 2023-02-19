import numpy as np
from numpy.fft import fft, fftshift, ifft, ifftshift
from matplotlib import pyplot as plt

def gauss_1d(x, mu, sigma):
    return np.exp(-((x - mu) * (x - mu) / (2 * sigma * sigma)))

def test_sum(n_samples, n_cycles=1, a=1, mu=None, sigma=0.1):
    x = np.arange(n_samples)
    if mu == None:
        mu = n_samples/2
    freq = n_cycles / n_samples
    out = 0.5*a*(np.cos(2*np.pi * x * freq))**2 + 0.5*a*gauss_1d(x, mu, n_samples * sigma)
    return out

def intensity(field):
    return (np.abs(field)**2)/256

def mse(a, b):
    return np.mean((a - b)**2)

def image(field, setup):
    '''
    Returns the image of a field through a 4F lens setup.
    inputs:
        field: 1D array of complex field values
        setup: dictionary of system parameters
            setup['N']: number of points in field
            setup['L']: field length in meters
            setup['lambda_0']: wavelength in meters
            setup['f1']: focal length of first lens in meters
            setup['f2']: focal length of second lens in meters
            setup['A']: aperture diameter in meters
            setup['plot']: boolean, whether to plot the image intensity
    returns:
        1D array of complex field values
    '''
    
    
    N = setup['N']
    L = setup['L']
    lambda_0 = setup['lambda_0']
    f1 = setup['f1']
    A = setup['A']
    f2 = setup['f2']
    plot = setup['plot']
    
    # compute coordinates for aperture field
    kx = np.fft.fftfreq(N, d=L/N)
    kx = np.fft.fftshift(kx)*lambda_0*f1

    # length of the field at the aperture
    L_a = np.max(kx) - np.min(kx)
    
    # number of samples for the aperture: (A / L_a) * N
    A_samples = int((A / L_a) * N)
    # mask consists of A_samples ones in the middle of a bed of N zeros
    n_zeros = int((N - A_samples)/2)
    mask = np.concatenate((np.zeros(n_zeros), np.ones(A_samples), np.zeros(n_zeros)))
    
    # compute aperture field and apply mask
    A_field = fftshift(fft(field, norm="ortho"))
    masked = A_field * mask
    
    # compute coordinates for image field
    kx_o = np.fft.fftfreq(N, d=L_a/N)
    kx_o = np.fft.fftshift(kx_o)*lambda_0*f2
    kx_o_mm = kx_o * 1000 # mm
    
    out = ifft(ifftshift(masked), norm='ortho')
    
    if plot:
        x_mm = np.linspace(-L/2, L/2, N) * 1000
        kx_mm = kx * 1000 # mm
        _, axs = plt.subplots(5, 1, figsize=(20, 15))
        axs[0].plot(x_mm, intensity(field))
        axs[0].set_xlabel('input field, mm')
        axs[1].plot(kx_mm, intensity(A_field))
        axs[1].set_xlabel('aperture field, mm')
        axs[2].plot(kx_mm, mask)
        axs[2].set_xlabel('mask, mm')
        axs[3].plot(kx_mm, intensity(masked))
        axs[3].set_xlabel('aperture field (masked), mm')
        axs[4].plot(kx_o_mm, intensity(out))
        axs[4].set_xlabel('output field, mm')
    
    return out
    
