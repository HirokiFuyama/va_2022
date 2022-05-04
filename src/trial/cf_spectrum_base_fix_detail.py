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
    time_window: int = 4  # day
    slide_window: int = 1  # day
    time_len: int = 15  # sec
    bins_freq: int = 50  # Hz
    freq_th_low: int = 100  # Hz
    freq_th_high: int = 3500  # Hz

    pre_base_start_date: str = "20210701"  # YYYYMMDD
    pre_base_end_date: str = "20210730"  # YYYYMMDD

    post_base_start_date: str = "20211215"
    post_base_end_date: str = "20211225"

    pre_target_start_date: str = "20210801"  # YYYYMMDD
    pre_target_end_date: str = "20221212"  # YYYYMMDD

    post_target_start_date: str = "20211226"
    post_target_end_date: str = "20220131"

    remove_date: str = "20211217"  # YYYYMMDD
    a_path: str = '../../data/raw/A/*.wav'
    v_path: str = '../../data/raw/V/*.wav'
    save_path: str = '../../figure/test.html'


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


def mk_path_list(path_list, base_start_day, base_end_day, target_start_day, target_end_day):
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
        (df["date"] >= pd.to_datetime(Config.pre_base_start_date)) & (df['date'] <= pd.to_datetime(Config.pre_base_end_date))
        ]['path'].values

    target_df = df[
        (df["date"] >= pd.to_datetime(Config.pre_target_start_date)) & (df['date'] <= pd.to_datetime(Config.pre_target_end_date))
        ]

    target_df = target_df[target_df['date'] != pd.to_datetime(Config.remove_date)]
    target_path_array = target_df['path'].values
    target_date_list = target_df['date'].astype("str").tolist()

    print("Base file n = {}".format(len(base_path_array)))
    print("Target file n = {}".format(len(target_path_array)))

    return base_path_array, target_path_array, target_date_list


def fft_diff(base_power_array, target_path_list):
    """

    Args:
        base_power_array (ndarray):
        target_path_list (list):

    Returns:

    """

    power_diff_list = []
    # wav_path_list = []
    for i in range(len(target_path_list)):
        wav_array = read_wav(target_path_list[i])
        fft = stft.Fft(wav_array)
        power, _ = fft.fft_cut_bin(
            freq_low=Config.freq_th_low,
            freq_high=Config.freq_th_high,
            bins_hz=Config.bins_freq,
        )

        power_diff_list.append(base_power_array-power)
        # wav_path_list.append(target_path_list[i])

        # if i % 10 == 0:
        #     print(i, '/', len(target_path_list))
    return power_diff_list


def fft_cut_bin_time_window(power_list, date_list, time_win=Config.time_window):
    """
    PTA前後で分離して、時間（日付）方向に平均を取る

    Args:
        power_list
        date_list (list): list of str
        time_win (int): unit: day

    Returns:

    """
    # split
    pta_date_idx = date_list.index("2021-12-17")

    pre_power_list = power_list[:pta_date_idx]
    post_power_list = power_list[pta_date_idx:]

    pre_date_list = date_list[:pta_date_idx]
    post_date_list = date_list[pta_date_idx:]

    pre_power_win = [
        np.mean(pre_power_list[i:i + time_win], axis=0) for i in range(0, len(pre_power_list) - time_win, time_win)
    ]

    post_power_win = [
        np.mean(post_power_list[i:i + time_win], axis=0) for i in range(0, len(post_power_list) - time_win, time_win)
    ]

    pre_date_win = [
        pre_date_list[i] + " - " + pre_date_list[i + time_win] for i in
        range(0, len(pre_date_list) - time_win, time_win)
    ]

    post_date_win = [
        post_date_list[i] + " - " + post_date_list[i + time_win] for i in
        range(0, len(post_date_list) - time_win, time_win)
    ]

    power_time_win = pre_power_win + post_power_win
    date_list_win = pre_date_win + post_date_win

    return np.array(power_time_win), date_list_win


def check_after_pta(power_list, date_list, time_win=Config.time_window, slide_win=Config.slide_window):
    """
    PTA後の周波数を見る

    Args:
        power_list:
        date_list:
        time_win:
        slide_win:

    Returns:

    """

    power_win = [
        np.average(power_list[i - slide_win: i], axis=0, weights=np.arange(1, slide_win + 1))
        for i in range(time_win, len(power_list), slide_win)
    ]

    return power_win, date_list[time_win:]


def fft_cut_bin_time_window_overlap(power_list, date_list, time_win=Config.time_window, slide_win=Config.slide_window):
    """
    PTA前後で分離して、時間（日付）方向に平均を取る

    Args:
        power_list
        date_list (list): list of str
        time_win (int): unit: day

    Returns:

    """
    # split
    pta_date_idx = date_list.index("2021-12-13")

    pre_power_list = power_list[:pta_date_idx]
    post_power_list = power_list[pta_date_idx:]

    pre_date_list = date_list[time_win: pta_date_idx]
    post_date_list = date_list[pta_date_idx + time_win:]

    pre_power_win = [
        np.average(pre_power_list[i-slide_win: i], axis=0, weights=np.arange(1, slide_win+1))
        for i in range(time_win, len(pre_power_list), slide_win)
    ]

    post_power_win = [
        np.average(post_power_list[i-slide_win: i], axis=0, weights=np.arange(1, slide_win+1))
        for i in range(time_win, len(post_power_list), slide_win)
    ]

    power_time_win = pre_power_win + post_power_win
    date_list_win = pre_date_list + post_date_list

    return np.array(power_time_win), date_list_win


def plotly_heatmap(power_win, date_win, base_freq, binary=False, save_path=Config.save_path):
    """

    Args:
        power_win:
        date_win:
        base_freq:
        save_path:

    Returns:

    """
    df = pd.DataFrame(power_win)
    df.columns = [int(i) for i in base_freq]
    df.index = [i + "_" for i in date_win]

    if binary:
        df = df.applymap(lambda x: 1 if x > 0 else 0)
    else:
        pass

    fig = go.Figure(data=go.Heatmap(
        z=df.T.values,
        x=df.index,
        y=df.columns,
    ))
    # colorscale='Viridis'))
    offline.plot(fig, filename=save_path, auto_open=True)


def process_two_base(glob_path):

    path_list = glob.glob(glob_path)
    path_list.sort()
    base_path_list, target_path_list, target_date_list = mk_path_list(path_list)



def process(glob_path):

    path_list = glob.glob(glob_path)
    path_list.sort()
    base_path_list, target_path_list, target_date_list = mk_path_list(path_list)

    #
    base_power, base_freq = base_power_spectrum(base_path_list)

    #
    power_diff_list = fft_diff(base_power, target_path_list)

    # moving average
    power_win, date_win = fft_cut_bin_time_window_overlap(power_diff_list, target_date_list)
    # power_win, date_win = fft_cut_bin_time_window(power_win, date_win)

    # check after pta
    # power_win, date_win = check_after_pta(power_diff_list, target_date_list)

    # plot
    plotly_heatmap(power_win, date_win, base_freq, binary=True)


def main():
    process(Config.a_path)
    # process(Config.v_path)


if __name__ == "__main__":
    main()
