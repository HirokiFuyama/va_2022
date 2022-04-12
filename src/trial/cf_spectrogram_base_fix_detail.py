import glob
from dataclasses import dataclass

import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
import matplotlib.pyplot as plt
import plotly.offline as offline
import plotly.graph_objects as go

import src.util.stft as stft


@dataclass
class Config:
    # a_path: str = rf'G:\マイドライブ\A\*.wav'
    # v_path: str = rf'G:\マイドライブ\V\*.wav'
    stft_window: int = 2  # sec
    stft_slide: int = 1  # sec
    time_window: int = 5  # day
    time_len: int = 15  # sec
    bins_freq: int = 50  # Hz
    freq_th_low: int = 500  # Hz
    freq_th_high: int = 2000  # Hz
    base_start_date: str = "20210101"  # YYYYMMDD
    base_end_date: str = "20210228"  # YYYYMMDD
    target_start_date: str = "20211201"  # YYYYMMDD
    target_end_date: str = "20211230"  # YYYYMMDD
    a_path: str = '../../data/raw/A/*.wav'
    v_path: str = '../../data/raw/V/*.wav'


def read_wav(file_path):
    """

    Args:
        file_path (str):

    Returns:
        np.array

    """
    fs = 44100
    sig = sf.read(file_path)[0]
    center_idx = int(len(sig) / 2)
    sig = sig[center_idx - fs * Config.time_len: center_idx + fs * Config.time_len]
    return sig


def base_spectrogram(base_path_list):
    """

    Args:
        base_path_list:

    Returns:
        pd.DataFrame: index=frequency(Hz), columns=time(sec), values=power spectrum(dB)

    """
    power = []
    for _path in base_path_list:
        _stft = stft.Stft(
            wav_array=read_wav(_path),
            window=Config.stft_window,
            slide=Config.stft_slide,
            freq_low=Config.freq_th_low,
            freq_high=Config.freq_th_high
        )

        _power, _freq, _time = _stft.stft_bin(bins_hz=Config.bins_freq)
        power.append(_power)
    power_mean = np.mean(power, axis=0)
    df = pd.DataFrame(power_mean)
    df.columns = [int(i) for i in _time]
    df.index = [int(i) for i in _freq]
    return df


def mk_path_list(path_list):
    """

    Args:
        path_list:

    Returns:

    """
    df = pd.DataFrame(path_list, columns=["path"])
    df["date"] = [i.split("/")[-1][:8] for i in df['path']]
    # df["date"] = [i.split("\\")[-1][:8] for i in df['path']]
    df["date"] = pd.to_datetime(df["date"])

    base_path_array = df[
        (df["date"] >= pd.to_datetime(Config.base_start_date)) & (df['date'] <= pd.to_datetime(Config.base_end_date))
        ]['path'].values

    target_df = df[
        (df["date"] >= pd.to_datetime(Config.target_start_date)) & (df['date'] <= pd.to_datetime(Config.target_end_date))
        ]

    target_path_array = target_df['path'].values
    target_date_list = target_df['date'].astype("str").tolist()

    return base_path_array, target_path_array, target_date_list


# def min_max_normalization(diff_power_array):
#     """
#
#     Args:
#         diff_power_array (ndarray): 3 dimension
#
#     Returns:
#
#     """
#     _max = diff_power_array.max()
#     _min = diff_power_array.min()
#     _diff = _max - _min
#
#     diff_power_array = (diff_power_array-_min)/_diff
#     return diff_power_array



# def fft_cut_bin_time_window(power_list, date_list, time_win=Config.time_window):
#     """
#
#     Args:
#         power_list
#         date_list (list): list of str
#         time_win (int): unit: day
#
#     Returns:
#
#     """
#
#     power_time_window = [
#         np.mean(power_list[i:i + time_win], axis=0) for i in range(0, len(power_list) - time_win, time_win)
#     ]
#
#     date_list_window = [
#         date_list[i] + " - " +  date_list[i + time_win] for i in range(0, len(date_list) - time_win, time_win)
#     ]
#     return np.array(power_time_window), date_list_window


def process(glob_path):

    path_list = glob.glob(glob_path)
    path_list.sort()
    base_path_list, target_path_list, target_date_list = mk_path_list(path_list)

    base_df = base_spectrogram(base_path_list)

    power_diff_list = []
    wav_path_list = []
    for i in range(len(target_path_list)):
        _stft = stft.Stft(
            wav_array=read_wav(target_path_list[i]),
            window=Config.stft_window,
            slide=Config.stft_slide,
            freq_low=Config.freq_th_low,
            freq_high=Config.freq_th_high
        )
        _power, _, _ = _stft.stft_bin(bins_hz=Config.bins_freq)

        power_diff_list.append(base_df.values - _power)
        wav_path_list.append(target_path_list[i])

        if i % 10 == 0:
            print(i, '/', len(target_path_list))

    # 周波数のみbin
    plotly_scatter = []
    power_diff_list = np.array(power_diff_list)
    for n, i in enumerate(power_diff_list):
        _df = pd.DataFrame(i)
        _df.index = base_df.index
        _df.columns = base_df.columns

        _scatter = go.Scatter(x=_df.columns.to_list(), y=_df.loc[600, :].values)
        plotly_scatter.append(_scatter)

    fig = go.Figure(data=plotly_scatter)
    offline.plot(fig, filename="../../figure/test", auto_open=True)
    # plt.legend(fontsize=15, loc="upper left")
    # plt.show()

    # 周波数、日時共にbin
    # power_win, date_win = fft_cut_bin_time_window(power_diff_list, target_date_list)
    # df = pd.DataFrame(power_win)
    # df.columns = [int(i) for i in base_freq]
    # df.index = date_win
    # plt.figure(figsize=(20, 20))
    # sns.heatmap(df)
    # plt.show()

    # 周波数、日時binを日時で平均
    # power_win_mean = np.mean(power_win, axis=1)
    # fig = go.Figure([go.Scatter(x=date_win, y=power_win_mean)])
    # offline.plot(fig, filename="../../figure/test", auto_open=True)


def main():
    process(Config.a_path)
    # process(Config.v_path)


if __name__ == "__main__":
    main()
