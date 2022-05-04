import numpy as np
from scipy import signal
import plotly.offline as offline
import plotly.graph_objects as go

from src.util.stft import Fft


class Filter(Fft):

    def __init__(self, wav_array, save_dir_path, fs=44100):
        """

        Args:
            wav_array (array like):
            save_dir_path (str): Path of html figure of frequency response
            fs (int): Sampling Frequency (Hz)
        """
        super().__init__(wav_array, fs)
        self.save_path = save_dir_path

    def _plot_frequency_response(self, b1, a1):
        """Frequency response of filter

        Args:
            b1:
            a1:

        Returns:

        """

        w1, h1 = signal.freqz(b1, a1, self.fs)
        frequency = w1 / (2 * np.pi) * self.fs
        power = 20 * np.log10(abs(h1))

        scatter = go.Scatter(x=frequency, y=power)
        fig = go.Figure(data=scatter)

        fig.update_layout(
            title="Frequency response of filter",
            xaxis_title="Frequency [Hz]",
            yaxis_title="Power [dB]",
        )

        save_path = self.save_path + "response.html"
        offline.plot(fig, filename=save_path, auto_open=True)

    def _plot_power_spectrum(self, b1, a1):
        """ Compare original average spectrum with filtered average spectrum

        Args:
            b1:
            a1:

        Returns:

        """
        raw_power, raw_freq = self.fft()

        filtered_wav_array = signal.filtfilt(b1, a1, self.wav_array)
        filtered_power, filtered_freq = Fft(filtered_wav_array).fft()

        scatter1 = go.Scatter(x=raw_freq, y=raw_power, name="Raw")
        scatter2 = go.Scatter(x=filtered_freq, y=filtered_power, name="Filtered")
        fig = go.Figure(data=[scatter1, scatter2])

        fig.update_layout(
            title="Average of Spectrum",
            xaxis_title="Frequency [Hz]",
            yaxis_title="Power [dB]",
        )

        save_path = self.save_path + "compare_ave_spectrum.html"
        offline.plot(fig, filename=save_path, auto_open=True)

    def _plot_time_waveform(self, b1, a1):
        """ Compare original time signal with filtered time signal

        Args:
            b1:
            a1:

        Returns:

        """
        filtered_wav_array = signal.filtfilt(b1, a1, self.wav_array)
        time_array = np.linspace(0, int(len(self.wav_array)/self.fs), len(self.wav_array))

        scatter1 = go.Scatter(x=time_array, y=self.wav_array, name="Raw")
        scatter2 = go.Scatter(x=time_array, y=filtered_wav_array, name="Filtered")
        fig = go.Figure(data=[scatter1, scatter2])

        fig.update_layout(
            title="Time waveform",
            xaxis_title="Time(s)",
            yaxis_title="Amplitude",
        )

        save_path = self.save_path + "compare_time_waveform.html"
        offline.plot(fig, filename=save_path, auto_open=True)

    def band_pass(self, fp1, fp2, fp_s1, fp_s2, gstop, gpass=1, plot_response=True):
        """

        Args:
            fp1 (float): Pass-through low end frequency [Hz]
            fp2 (float): Pass-through high end frequency [Hz]
            fp_s1 (float): Blocking low end frequency [Hz]
            fp_s2 (float): Blocking high end frequency[Hz]
            gstop (float): Minimum loss at the blocking side [dB]
            gpass (float): Maximum loss at the pass side [dB]
            plot_response (bool): Plot of frequency response

        Returns:

        """
        wp1 = fp1 / (self.fs / 2)
        wp2 = fp2 / (self.fs / 2)
        ws1 = fp_s1 / (self.fs / 2)
        ws2 = fp_s2 / (self.fs / 2)

        n1, wn1 = signal.buttord([wp1, wp2], [ws1, ws2], gpass, gstop)
        b1, a1 = signal.butter(n1, wn1, "bandpass")

        if plot_response:
            self._plot_frequency_response(b1, a1)
            self._plot_power_spectrum(b1, a1)
            self._plot_time_waveform(b1, a1)

        return signal.filtfilt(b1, a1, self.wav_array)

    def band_stop(self, fp1, fp2, fp_s1, fp_s2, gstop, gpass=1, plot_response=True):
        """

        Args:
            fp1 (float): Pass-through low end frequency [Hz]
            fp2 (float): Pass-through high end frequency [Hz]
            fp_s1 (float): Blocking low end frequency [Hz]
            fp_s2 (float): Blocking high end frequency[Hz]
            gstop (float): Minimum loss at the blocking side [dB]
            gpass (float): Maximum loss at the pass side [dB]
            plot_response (bool): Plot of frequency response

        Returns:

        """
        wp1 = fp1 / (self.fs / 2)
        wp2 = fp2 / (self.fs / 2)
        ws1 = fp_s1 / (self.fs / 2)
        ws2 = fp_s2 / (self.fs / 2)

        n1, wn1 = signal.buttord([wp1, wp2], [ws1, ws2], gpass, gstop)
        b1, a1 = signal.butter(n1, wn1, "bandstop")

        if plot_response:
            self._plot_frequency_response(b1, a1)
            self._plot_power_spectrum(b1, a1)
            self._plot_time_waveform(b1, a1)

        return signal.filtfilt(b1, a1, self.wav_array)

    def low_pass(self, fp, fp_s, gstop, gpass=1, plot_response=True):
        """

        Args:
            fp (float): Pass-through end frequency [Hz]
            fp_s (float): Blocking end frequency [Hz]
            gstop (float): Minimum loss at the blocking side [dB]
            gpass (float): Maximum loss at the pass side [dB]
            plot_response (bool): Plot of frequency response

        Returns:

        """

        wp = fp / (self.fs / 2)
        ws = fp_s / (self.fs / 2)

        nd, wn = signal.buttord(wp, ws, gpass, gstop)
        b1, a1 = signal.butter(nd, wn, "low")

        if plot_response:
            self._plot_frequency_response(b1, a1)
            self._plot_power_spectrum(b1, a1)
            self._plot_time_waveform(b1, a1)

        return signal.filtfilt(b1, a1, self.wav_array)

    def high_pass(self, fp, fp_s, gstop, gpass=1, plot_response=True):
        """

        Args:
            fp (float): Pass-through end frequency [Hz]
            fp_s (float): Blocking end frequency [Hz]
            gstop (float): Minimum loss at the blocking side [dB]
            gpass (float): Maximum loss at the pass side [dB]
            plot_response (bool): Plot of frequency response

        Returns:

        """

        wp = fp / (self.fs / 2)
        ws = fp_s / (self.fs / 2)

        nd, wn = signal.buttord(wp, ws, gpass, gstop)
        b1, a1 = signal.butter(nd, wn, "high")

        if plot_response:
            self._plot_frequency_response(b1, a1)
            self._plot_power_spectrum(b1, a1)
            self._plot_time_waveform(b1, a1)

        return signal.filtfilt(b1, a1, self.wav_array)
