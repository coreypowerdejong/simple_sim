import numpy as np
from numpy.fft import fft, fftshift, ifft, ifftshift
from matplotlib import pyplot as plt

golomb = np.array([1, 4, 10, 12, 17]) # Golomb ruler sequence

def gauss_1d(x, mu, sigma):
    return np.exp(-((x - mu) * (x - mu) / (2 * sigma * sigma)))

def gauss_1d_fwhm(x, mu, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return gauss_1d(x, mu, sigma)

def test_sum(n_samples, n_cycles=1, a=1, mu=None, sigma=0.1):
    x = np.arange(n_samples, dtype=np.complex128)
    if mu == None:
        mu = n_samples/2
        
    freq = n_cycles / n_samples * golomb/golomb[-1]
    out = np.zeros(n_samples, dtype=np.complex128)
    for i, f in enumerate(freq):
        out += (0.5/(i+1))*a*(np.cos(2*np.pi * x * f))**2 # amplitude gets lower as frequency gets higher
    out += a*gauss_1d(x, mu, n_samples * sigma)
    return out

def intensity(field):
    return (np.abs(field)**2)/256

def mse(a, b):
    return np.mean((a - b)**2)

def center_splice(destination, source):
    """Replace the center of destination with source."""
    d = len(destination)
    s = len(source)
    destination[d//2-s//2:d//2+s//2] = source
    return destination
    
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
    if A_samples % 2 == 1:
        A_samples += 1
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
    
def led_array(n_leds=32, n_samples=5000, size=223e-3, spacing=4e-3, fwhm=0.1e-3, a=1):
    '''
    Returns an array of LED fields.
    inputs:
        n_leds: number of LEDs
        n_samples: number of samples in each LED field
        size: size of the LED array in meters
        spacing: spacing between LEDs in meters
        fwhm: full width at half maximum of the LED field in meters
        a: amplitude of the LED field
    returns:
        2D array of complex field values
    '''
    x = np.arange(n_samples, dtype=np.complex128)
    w_samples = int((fwhm / size) * n_samples)
    center = int(n_samples/2)
    positions = [center] + [center + ((-1)**(i+1)*np.ceil(i/2)*spacing*n_samples/(2*size)) for i in range(1, n_leds)]
    for mu in positions: # positions are modelled as the mean of the gaussian
        out = a*gauss_1d_fwhm(x, mu, w_samples)
        yield out 

def propogate_field(x_in, n_samples=5000, size=223e-3, lambda_0=650e-9, z=0.08):
    # compute coordinates for x_out
    kx = np.fft.fftfreq(n_samples, d=size/n_samples) # spatial frequencies
    kx = np.fft.fftshift(kx)*lambda_0*z
    
    # propogate field using fourier transform
    x_out = fftshift(fft(x_in))
    
    return x_out, kx
