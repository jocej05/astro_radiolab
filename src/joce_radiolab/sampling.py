import numpy as np

def load_npz(filename):
    """
    Load SDR data saved as .npz.
    Expected keys: 'data', 'sample_rate'
    """
    z = np.load(filename)
    return z["data"], float(z["sample_rate"])


def power_spectrum(x, fs, shift=True):
    """
    Compute power spectrum |FFT|^2.

    Parameters
    ----------
    x : array_like
        Time-domain samples
    fs : float
        Sampling rate (Hz)
    shift : bool
        If True, return fftshifted spectrum

    Returns
    -------
    freqs : ndarray
        Frequency axis (Hz)
    P : ndarray
        Power spectrum (arb. units)
    """
    x = np.asarray(x)
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(len(x), d=1/fs)
    P = np.abs(X)**2

    if shift:
        freqs = np.fft.fftshift(freqs)
        P = np.fft.fftshift(P)

    return freqs, P
