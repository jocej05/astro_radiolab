import numpy as np
from joce_radiolab.sampling import power_spectrum


def test_fft_peak_for_sine():
    """
    Verify that the FFT power spectrum correctly identifies
    the frequency of a pure sine wave.
    """
    fs = 1_000_000      # sample rate (Hz)
    f0 = 300_000        # sine wave frequency (Hz)
    N = 16384           # number of samples

    t = np.arange(N) / fs
    x = np.sin(2 * np.pi * f0 * t)

    freqs, P = power_spectrum(x, fs, shift=False)
    f_peak = freqs[np.argmax(P)]

    # Peak should occur at ±f0 (FFT symmetry)
    assert np.isclose(abs(f_peak), f0, atol=fs / N)
