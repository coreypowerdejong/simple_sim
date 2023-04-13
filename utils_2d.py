import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift, fft
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import utils
import cv2 as cv

golomb = np.array([1, 4, 10, 12, 17]) # Golomb ruler sequence

def gauss_2d(x, y, mu_x, mu_y, sigma_x, sigma_y):
    return utils.gauss_1d(x, mu_x, sigma_x) * utils.gauss_1d(y, mu_y, sigma_y)

def gauss_2d_fwhm(x, y, mu_x, mu_y, fwhm_x, fwhm_y):
    sigma_x = fwhm_x / (2 * np.sqrt(2 * np.log(2)))
    sigma_y = fwhm_y / (2 * np.sqrt(2 * np.log(2)))
    return gauss_2d(x, y, mu_x, mu_y, sigma_x, sigma_y)

def center_splice(destination, source):
    """Replace the center of destination with source."""
    d = len(destination)
    s = len(source)
    destination[d//2-s//2:d//2+s//2] = source
    return destination

def center_splice_2d(destination, source):
    """Replace the center of 2D destination with source."""

    dx = destination.shape[0]
    dy = destination.shape[1]
    
    sx = source.shape[0]
    sy = source.shape[1]
    
    destination[dx//2-sx//2:dx//2+sx//2, dy//2-sy//2:dy//2+sy//2] = source
    return destination

def center_cut_2d(source, size, center=None):
    """Return the center square of 2D source."""
     
    sx = source.shape[0]
    sy = source.shape[1]
    if center == None:
        center = (sx//2, sy//2)
    
    return source[int(center[0] - size//2):int(center[0] + size//2), int(center[1] - size//2):int(center[1] + size//2)]

def get_circular_mask(N, size, center, radius):
    """
    Return a circular mask of size N with radius and center.
    inputs:
        N: number of points in each axis of mask
        size: size of mask in meters
        center: tuple of (x, y) center of circle in meters
        radius: radius of circle in meters
    returns:
        Indices of the N x N mask that are within the circle, suitable for indexing.
    """
    
    x = np.linspace(-size/2, size/2, N)
    y = np.linspace(-size/2, size/2, N)
    xx, yy = np.meshgrid(x, y)
    mask = (xx - center[0])**2 + (yy - center[1])**2 <= radius**2
    return np.nonzero(mask)

def image_2d(field, setup):
    '''
    Returns the image of a field through a 4F lens setup.
    The input field must be a square matrix.
    inputs:
        field: 2D array of complex field values
        setup: dictionary of system parameters
            setup['N']: number of points in each axis field
            setup['L']: field length in meters
            setup['lambda_0']: wavelength in meters
            setup['f1']: focal length of first lens in meters
            setup['f2']: focal length of second lens in meters
            setup['A']: aperture diameter in meters
            setup['plot']: boolean, whether to plot the image intensity
    returns:
        2D array of complex field values
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
    
    # select sample indices for the aperture
    r_A_samples = int((A / L_a) * N / 2)
    A_indices = get_circular_mask(N, L_a, (0, 0), A/2)
    
    # mask consists of circular section of the aperture field
    # compute aperture field and apply mask
    A_field = fftshift(fft2(fftshift(field), norm="ortho"))
    masked = np.zeros((N, N), dtype=np.complex128)
    masked[A_indices] = A_field[A_indices]
    masked_cutout = center_cut_2d(masked, 2*r_A_samples)
    
    # compute coordinates for image field
    kx_o = np.fft.fftfreq(len(kx), d=A/len(kx))
    kx_o = np.fft.fftshift(kx_o)*lambda_0*f2
    kx_o_mm = kx_o * 1000 # mm
    
    out = ifftshift(ifft2(ifftshift(masked_cutout), norm='ortho'))
    
    if plot:
        kx_mm = kx * 1000 # mm
        L_mm = L * 1000 # mm
        A_mm = A * 1000 # mm
        L_a_mm = L_a * 1000 # mm
        L_o_mm = np.max(kx_o_mm[N//2-r_A_samples:N//2+r_A_samples]) - np.min(kx_o_mm[N//2-r_A_samples:N//2+r_A_samples])

        _, axs = plt.subplots(2, 2, figsize=(15, 15))
        
        axs[0,0].imshow(utils.intensity(field), extent=[-L_mm/2, L_mm/2, -L_mm/2, L_mm/2], cmap='gray')
        axs[0,0].title.set_text('input field, mm')
        
        m = utils.intensity(A_field)
        axs[0,1].imshow(np.log(m, out=np.zeros_like(m), where=(m!=0)), extent=[-L_a_mm/2, L_a_mm/2, -L_a_mm/2, L_a_mm/2], cmap='gray')
        axs[0,1].title.set_text('aperture field (log intensity), mm')
        axs[0,1].add_patch(Circle((0, 0), A_mm/2, color='r', fill=False))
        
        n = utils.intensity(masked_cutout)
        axs[1,0].imshow(np.log(n, out=np.zeros_like(n), where=(n!=0)), extent=[-A_mm/2, A_mm/2, -A_mm/2, A_mm/2], cmap='gray')
        axs[1,0].title.set_text('aperture field (log intensity) (masked), mm')
        
        axs[1,1].imshow(utils.intensity(out), extent=[-L_o_mm/2, L_o_mm/2, -L_o_mm/2, L_o_mm/2], cmap='gray')
        axs[1,1].title.set_text('output field, mm')
    
    return out
    
def led_units(n_samples=5000, size=223e-3):
    return np.linspace(-size/2, size/2, n_samples)
    
def led_array_2d(n_leds=32, n_samples=5000, size=223e-3, spacing=4e-3, fwhm=0.1e-3, a=1):
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
    y = np.arange(n_samples, dtype=np.complex128)
    xx, yy = np.meshgrid(x, y)
    w_samples = int((fwhm / size) * n_samples)
    center = int(n_samples/2)
    index = np.array([(-1)**(i+1)*np.ceil(i/2) for i in range(1, n_leds)])
    index_2d = [(0,0)]
    for idx1 in index:
        for idx2 in index:
            index_2d.append((idx1, idx2))
            
    positions = [(center, center)] + ((center, center) + np.asarray(index_2d)*spacing*n_samples/(size)).tolist()
    for idx, (mu_x, mu_y) in zip(index_2d, positions): # positions are modelled as the mean of the gaussian
        out = a*gauss_2d_fwhm(xx, yy, mu_x, mu_y, w_samples, w_samples)
        yield idx, out

def propagate_fft_2d(x_in, n_samples=5000, size=223e-3, lambda_0=650e-9, z=0.08):
    """Propagates a 2D field using the fast fourier transform."""
    
    # compute coordinates for x_out
    kx = np.fft.fftfreq(n_samples, d=size/n_samples) # spatial frequencies
    kx = np.fft.fftshift(kx)*lambda_0*z
    
    # propogate field using fourier transform
    x_out = fftshift(fft2(fftshift(x_in)))
    
    return x_out, kx

def reconstruct_gs_2d(images, offset, target_resolution, n_iterations=3):
    """
    Reconstructs a high resolution image from a set of images using the Gerchberg-Saxton algorithm.
    Assmes that the propagation function is the Fourier transform (Fraunhofer regime).
    inputs:
        images: list of tuples (idx, img) where idx is the index of the illumination and img is the image generated by that illumination
        offset: offset between the illuminations (shift in position for cpm, or spectrum for fpm)
        target_resolution: resolution of the reconstructed image (pixels or samples)
        n_iterations: number of iterations to run the algorithm for
    returns:
        reconstructed image
    """
    
    # generate high resolution base object
    hr_object = cv.resize(images[0][1], (target_resolution, target_resolution), interpolation=cv.INTER_NEAREST)
    hr_spectrum = fftshift(fft(fftshift(hr_object)))
    
    r_A_indices = images[0][1].shape[0]//2
    # hr_aperture_indices = get_circular_mask(target_resolution, target_resolution, (target_resolution//2, target_resolution//2), r_A_indices)
    lr_aperture_indices = get_circular_mask(images[0][1].shape[0], images[0][1].shape[0], (images[0][1].shape[0]//2, images[0][1].shape[0]//2), r_A_indices)
    hr_aperture_indices = [lr_aperture_indices[0] + target_resolution//2 - r_A_indices, lr_aperture_indices[1] + target_resolution//2 - r_A_indices]
    sz = images[0][1].shape[0]
    for i in range(n_iterations):

        for (idx, img) in images:
            # step 1: generate low resolution target field from high resolution field spectrum according to band pass filter, shifted by plane wave illumination angle
            # sub_band = center_cut_2d(hr_spectrum, 2*r_A_indices, (sz//2 - int(idx[0]*offset), sz//2 - int(idx[1]*offset)))
            # sub_band = hr_spectrum[aperture_indices[0] - int(idx[0]*offset), aperture_indices[1] - int(idx[1]*offset)]
            sub_band = hr_spectrum[target_resolution//2 - r_A_indices - int(idx[0]*offset):target_resolution//2 + r_A_indices - int(idx[0]*offset), target_resolution//2 - r_A_indices - int(idx[1]*offset):target_resolution//2 + r_A_indices - int(idx[1]*offset)]
            
            target = ifftshift(ifft2(ifftshift(sub_band)))
            
            # step 2: generate low resolution measurement using image function
            # This is done in advance and passed in as an argument to this function

            # step 3: replace target amplitude with measurement amplitude
            a_lm, _ = utils.rect_to_polar(img)
            _, theta_l = utils.rect_to_polar(target)
            target = utils.polar_to_rect(a_lm, theta_l)
                    
            # step 4: replace high resolution spectrum band with updated target spectrum band
            hr_spectrum[hr_aperture_indices[0] - int(idx[0]*offset), hr_aperture_indices[1] - int(idx[1]*offset)] = fftshift(fft(fftshift(target)))[lr_aperture_indices]

    return hr_spectrum
