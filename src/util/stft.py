import numpy as np
from scipy import signal
from scipy import fftpack


def power_spectrum(data: np.ndarray, fs: int = 44100) -> np.ndarray:
    """ Calculate power spectrum

    zero padding at two times the length of data

    Args:
        data (np.ndarray): signal of sounds
        fs (int): sampling frequency

    Returns:
        power (np.ndarray): power spectrum
        freq (np.ndarray) : frequency
    """

    _n = len(data)
    # _n = int(len(data)*2)
    power = fftpack.fft(data, n=_n)
    power = abs(power)[0:round(_n / 2)] ** 2 / _n
    power = 10 * np.log10(power)
    freq = np.arange(0, fs / 2, fs / _n)

    return power, freq


def _hamming(data: np.ndarray) -> np.ndarray:
    """ FFT window function

    Args:
        data (np.ndarray): sound signal

    Returns:
        np.ndarray
    """
    win_hamming = signal.hamming(len(data))

    return data * win_hamming


def stft(data: np.ndarray, window: int = 10, slide: int = 1, fs: int = 44100):
    """ Short time FFT

    Args:
        data (np.ndarray):
        window (int): length of window (unit:sec)
        slide (int): length of slide window (unit:sec)
        fs (int): sampling frequency (unit:Hz)

    Returns:
        power (np.ndarray): power spectrum (unit:dB)
        freq (list): frequency (unit:sec)
        time (np.ndarray): time (unit:sec)
    """

    # th = -30  # db
    power = []
    freq = []
    for i in range(0, len(data)-int(window*fs), int(slide*fs)):
        p, f = power_spectrum(_hamming(data[i:i+int(window*fs)]), fs)
        power.append(p)
        freq.append(f)
    time = np.linspace(0, round(len(data)/fs), len(power))
    power = np.array(power).T
    # power = np.where(power>th, power, -100)

    return power, freq[0], time


def stft_bin(
        data: np.ndarray,
        bins: int = 10,
        window: int = 10,
        slide: int = 1,
        fs: int = 44100,
        freq_low: int = 30,
        freq_high: int = 1000
):
    """ Short time FFT

    Args:
        data (np.ndarray):
        bins (int): unit:Hz
        window (int): length of window (unit:sec)
        slide (int): length of slide window (unit:sec)
        fs (int): sampling frequency (unit:Hz)
        freq_low (int): unit:Hz
        freq_high (int): unit:Hz

    Returns:
        power (np.ndarray): power spectrum (unit:dB)
        freq (list): frequency (unit:sec)
        time (np.ndarray): time (unit:sec)
    """

    # th = -30  # db
    power = []
    for i in range(0, len(data)-int(window*fs), int(slide*fs)):
        p, f = power_spectrum(_hamming(data[i:i+int(window*fs)]), fs)

        f_unit_idx = int(bins / f[1])
        f_low_idx = int(freq_low / f[1])
        f_high_idx = int(freq_high / f[1])

        p = p[f_low_idx: f_high_idx]
        p = [p[i:i + f_unit_idx].mean() for i in range(0, len(p) - f_unit_idx, f_unit_idx)]

        power.append(p)
    freq = np.linspace(freq_low, freq_high, len(power[0]))
    time = np.linspace(0, round(len(data)/fs), len(power))
    power = np.array(power).T

    return power, freq, time