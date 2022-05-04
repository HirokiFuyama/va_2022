import numpy as np
from scipy import signal
from scipy import fftpack


class Fft:

    def __init__(self, wav_array, fs=44100):
        self.wav_array = wav_array
        self.fs = fs

    def fft(self, y_scale="dB", padding=False):
        """
        dB: Reference Sound Pressure = 20μPa (0dB)

        Args:
            y_scale (str): dB or liner
            padding (bool):

        Returns:

        """
        reference_pressure = 20e-6

        if padding:
            _n = int(len(self.wav_array)*2)
        else:
            _n = len(self.wav_array)
        power = fftpack.fft(self.wav_array, n=_n)
        power = abs(power)[0:round(_n / 2)] ** 2
        freq = np.arange(0, self.fs / 2, self.fs / _n)

        if y_scale == "dB":
            power = 20 * np.log10(power / reference_pressure)
        elif y_scale == "liner":
            pass

        return power, freq

    def fft_bin(self, bins_hz=1, padding=False):
        """Calculate the average value in units of frequency of bins_hz

        Args:
            bins_hz: int (unit:Hz)
            padding:

        Returns:

        """
        power, freq = self.fft(padding=padding)
        freq_unit_idx = int(bins_hz / freq[1])
        power_bin = [np.mean(power[i:i+freq_unit_idx]) for i in range(0, len(power)-freq_unit_idx, freq_unit_idx)]
        freq_bin = np.linspace(freq[0], freq[-1], len(power_bin))

        return power_bin, freq_bin

    def fft_cut(self, freq_low=30, freq_high=1000):
        """

        Args:
            freq_low:
            freq_high:

        Returns:

        """
        _power, _freq = self.fft()

        f_low_idx = int(freq_low / _freq[1])
        f_high_idx = int(freq_high / _freq[1])

        power = _power[f_low_idx: f_high_idx]
        freq = _freq[f_low_idx: f_high_idx]

        return power, freq

    def fft_cut_bin(self, freq_low=30, freq_high=1000, bins_hz=1):
        """

        Args:
            freq_low:
            freq_high:
            bins_hz:
        Returns:

        """
        power, freq = self.fft_cut(freq_low, freq_high)
        freq_unit_idx = int(bins_hz / (freq[1] - freq[0]))
        power_bin = [np.mean(power[i:i + freq_unit_idx]) for i in range(0, len(power) - freq_unit_idx, freq_unit_idx)]
        freq_bin = np.linspace(freq[0], freq[-1], len(power_bin))

        return power_bin, freq_bin


class Stft(Fft):

    def __init__(self, wav_array, fs=44100, window=10, slide=1, freq_low=30, freq_high=1000, fftpadding=False):
        """

        Args:
            wav_array:
            fs (int): sampling frequency (unit:Hz)
            window (int): length of window (unit:sec)
            slide (int): length of slide window (unit:sec)
            freq_low (int): unit:Hz
            freq_high (int): unit:Hz
        """
        super().__init__(wav_array, fs)
        self.window = window
        self.slide = slide
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.padding = fftpadding

    def _hamming(self, data):
        """ FFT window function

        Args:
            data (np.ndarray): wav_array

        Returns:
            np.ndarray
        """
        win_hamming = signal.windows.hamming(len(data))
        return data * win_hamming

    def _hann(self, data):
        """

        Args:
            data:

        Returns:

        """
        win_hann = signal.windows.hann(len(data))
        return data * win_hann

    def _flattop(self, data):
        """

        Returns:

        """
        win_flattop = signal.windows.flattop(len(data))
        return data * win_flattop

    def _fft(self, wav_array, padding=False):
        """fft

        Args:
            wav_array:
            padding:

        Returns:

        """
        if padding:
            _n = int(len(wav_array) * 2)
        else:
            _n = len(wav_array)
        power = fftpack.fft(wav_array, n=_n)
        power = abs(power)[0:round(_n / 2)] ** 2 / _n
        power = 10 * np.log10(power)
        freq = np.arange(0, self.fs / 2, self.fs / _n)
        return power, freq

    def stft(self, window_func="hamming"):
        """ Short time FFT

        Args:
            window_func (str): "hamming", "hann", "flattop"

        Returns:
            power (np.ndarray): power spectrum (unit:dB)
            freq (list): frequency (unit:sec)
            time (np.ndarray): time (unit:sec)
        """

        power = []
        freq = []
        for i in range(0, len(self.wav_array) - int(self.window * self.fs), int(self.slide * self.fs)):

            if window_func == "hann":
                p, f = self._fft(
                    self._hann(self.wav_array[i: i + int(self.window * self.fs)]),
                    padding=self.padding
                )

            elif window_func == "flattop":
                p, f = self._fft(
                    self._flattop(self.wav_array[i: i + int(self.window * self.fs)]),
                    padding=self.padding
                )

            else:
                p, f = self._fft(
                    self._hamming(self.wav_array[i: i + int(self.window * self.fs)]),
                    padding=self.padding
                )

            f_low_idx = int(self.freq_low / f[1])
            f_high_idx = int(self.freq_high / f[1])

            p = p[f_low_idx: f_high_idx]
            f = f[f_low_idx: f_high_idx]

            power.append(p)
            freq.append(f)

        time = np.arange(1, len(power)+1, 1)
        power = np.array(power).T

        return power, freq[0], time

    def stft_bin(self, bins_hz=10, window_func="hamming"):
        """ Short Time FFT

        Args:


        Returns:
            power (np.ndarray): power spectrum (unit:dB)
            freq (list): frequency (unit:sec)
            time (np.ndarray): time (unit:sec)
        """
        _power, _freq, time = self.stft(window_func)

        f_unit_idx = int(bins_hz / (_freq[1] - _freq[0]))

        power = []
        for _p in np.array(_power).T:
            power.append([np.mean(_p[i:i + f_unit_idx]) for i in range(0, len(_p) - f_unit_idx, f_unit_idx)])

        freq = np.arange(self.freq_low+bins_hz, self.freq_high, bins_hz)
        power = np.array(power).T

        return power, freq, time


class Cepstrum(Fft):

    def __init__(self, wav_array, fs=44100):
        super().__init__(wav_array, fs)

    def envelope(self, dim=500):
        """

        Args:
            dim (int): Number of samples for liftering

        Returns:

        """
        _n = len(self.wav_array)

        # FFT
        power = fftpack.fft(self.wav_array)
        power = abs(power) ** 2
        power = 20 * np.log10(power / 20e-6)

        # FFT -> IFFT
        cepstrum = fftpack.fft(power)
        cepstrum[dim: len(cepstrum) - dim] = 0
        envelope = np.real(fftpack.ifft(cepstrum))

        freq = np.arange(0, self.fs / 2, self.fs / _n)
        power = power[0:round(_n / 2)]
        envelope = envelope[0:round(_n / 2)]

        return power, freq, envelope
