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
    time_window: int = 5  # day
    time_len: int = 15  # sec
    bins_freq: int = 25  # Hz
    freq_th_low: int = 500  # Hz
    freq_th_high: int = 2000  # Hz
    base_start_date: str = "20210101"  # YYYYMMDD
    base_end_date: str = "20210228"  # YYYYMMDD
    target_start_date: str = "20210301"  # YYYYMMDD
    target_end_date: str = "20220131"  # YYYYMMDD
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


def base_power_spectrum(base_path_list):
    """

    Args:
        base_path_list:

    Returns:

    """
    power = []
    freq = []
    for _path in base_path_list:
        _fft = stft.Fft(read_wav(_path))
        _power, _freq = _fft.fft_cut_bin(
            freq_low=Config.freq_th_low,
            freq_high=Config.freq_th_high,
            bins_hz=Config.bins_freq
        )
        power.append(_power)
        freq.append(_freq)
    return np.mean(power, axis=0), freq[0]


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


def fft_cut_bin_time_window(power_list, date_list, time_win=Config.time_window):
    """

    Args:
        power_list
        date_list (list): list of str
        time_win (int): unit: day

    Returns:

    """

    power_time_window = [
        np.mean(power_list[i:i + time_win], axis=0) for i in range(0, len(power_list) - time_win, time_win)
    ]

    date_list_window = [
        date_list[i] + " - " +  date_list[i + time_win] for i in range(0, len(date_list) - time_win, time_win)
    ]
    return np.array(power_time_window), date_list_window


def process(glob_path):

    path_list = glob.glob(glob_path)
    path_list.sort()
    base_path_list, target_path_list, target_date_list = mk_path_list(path_list)

    base_power, base_freq = base_power_spectrum(base_path_list)

    power_diff_list = []
    wav_path_list = []
    for i in range(len(target_path_list)):
        wav_array = read_wav(target_path_list[i])
        fft = stft.Fft(wav_array)
        power, _ = fft.fft_cut_bin(
            freq_low=Config.freq_th_low,
            freq_high=Config.freq_th_high,
            bins_hz=Config.bins_freq,
        )

        power_diff_list.append(base_power-power)
        wav_path_list.append(target_path_list[i])

        if i % 10 == 0:
            print(i, '/', len(target_path_list))

    # 周波数のみbin
    # df = pd.DataFrame(power_diff_list)
    # df.columns = [int(i) for i in base_freq]
    # df.index = target_date_list
    # plt.figure(figsize=(20, 20))
    # sns.heatmap(df)
    # plt.show()

    # 周波数、日時共にbin
    power_win, date_win = fft_cut_bin_time_window(power_diff_list, target_date_list)
    # df = pd.DataFrame(power_win)
    # df.columns = [int(i) for i in base_freq]
    # df.index = date_win
    # plt.figure(figsize=(20, 20))
    # sns.heatmap(df)
    # plt.show()

    # 周波数、日時binを日時で平均
    power_win_mean = np.mean(power_win, axis=1)
    fig = go.Figure([go.Scatter(x=date_win, y=power_win_mean)])
    offline.plot(fig, filename="../../figure/test", auto_open=True)


def main():
    process(Config.a_path)
    # process(Config.v_path)


if __name__ == "__main__":
    main()
