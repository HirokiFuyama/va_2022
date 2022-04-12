import glob
from dataclasses import dataclass

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import mean_squared_error

import src.util.stft as stft
import src.util.distinguish_av as distinguish_av


@dataclass
class Config:
    # a_path: str = rf'G:\マイドライブ\A\*.wav'
    # v_path: str = rf'G:\マイドライブ\V\*.wav'
    time_len: int = 15  # sec
    freq_th: int = 1000  # Hz
    base_year: int = 2021
    base_month: int = 1
    a_path: str = '/Volumes/GoogleDrive-110826503708205819846/マイドライブ/A/*.wav'
    v_path: str = '/Volumes/GoogleDrive-110826503708205819846/マイドライブ/V/*.wav'


def calculate_rmse(window_path_list, wav_path):
    """

    Args:
        window_path_list:
        wav_path:

    Returns:

    """

    fs = 44100

    base_power, base_freq = distinguish_av.mean_spectrum(
        window_path_list,
        fs=fs,
        time_len=Config.time_len,
        threshold=Config.freq_th
    )

    sig = sf.read(wav_path)[0]
    center_idx = int(len(sig) / 2)
    sig = sig[center_idx - fs * Config.time_len: center_idx + fs * Config.time_len]
    power, freq = stft.power_spectrum(sig)
    th_idx = int(Config.freq_th / freq[1])
    power = power[:th_idx]

    if len(power) == len(base_power):
        rmse = np.sqrt(mean_squared_error(base_power, power))

    else:
        rmse = np.nan

    return rmse


def mk_path_list(path_list):
    """

    Args:
        path_list:

    Returns:

    """
    df = pd.DataFrame(path_list, columns=["path"])
    df["date"] = [i.split("\\")[-1][:8] for i in df['path']]
    df["date"] = pd.to_datetime(df["date"])
    df['year'] = [i.year for i in df["date"]]
    df['month'] = [i.month for i in df["date"]]

    base_df = df[df['year'] == Config.base_year]
    base_path_list = base_df[base_df['month'] == Config.base_month]['path'].values

    path_list = base_df[base_df['month'] >= Config.base_month+1]['path'].values

    return base_path_list, path_list


def process(glob_path, save_path):

    path_list = glob.glob(glob_path)
    path_list.sort()
    base_path_list, path_list = mk_path_list(path_list)

    rmse_list = []
    wav_path_list = []
    for i in range(len(path_list)):
        wav_path = path_list[i]

        _rmse = calculate_rmse(base_path_list, wav_path)

        rmse_list.append(_rmse)
        wav_path_list.append(wav_path)

        if i % 10 == 0:
            print(i, '/', len(path_list))

    df = pd.DataFrame(rmse_list, columns=['rmse'])
    df['path'] = wav_path_list
    return df.to_csv(save_path, index=False)


def main():
    process(Config.a_path, "../../data/process/fix_a_1000.csv")
    process(Config.v_path, "../../data/process/fix_v_1000.csv")


main()
