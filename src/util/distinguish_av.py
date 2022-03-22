import glob
import shutil
from dataclasses import dataclass

import numpy as np
import soundfile as sf
# import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import src.util.stft as stft


@dataclass
class Config:
    a_path: str = rf"C:\Users\HirokiFuruyama\analysis\va_2021\data\A\*"
    v_path: str = rf"C:\Users\HirokiFuruyama\analysis\va_2021\data\V\*"
    root_dir_path: str = rf'G:\マイドライブ'
    wav_dir_path: str = rf'G:\マイドライブ\シャント\*.wav'


def mean_spectrum(path_list, fs=44100, time_len=10, threshold=5000):
    """

    Args:
        path_list (list):
        fs (int):  Hz
        time_len (int): sec
        threshold (int): Hz

    Returns:

    """
    freq = []
    power = []
    for i in path_list:
        sig = sf.read(i)[0]

        center_idx = int(len(sig)/2)
        sig = sig[center_idx - fs*time_len: center_idx + fs*time_len]

        p, f = stft.power_spectrum(sig)

        th_idx = int(threshold/f[1])
        power.append(p[:th_idx])
        freq.append(f[:th_idx])

    power = np.array(power).mean(axis=0)
    freq = freq[0]

    return power, freq


def process(wav_path, fs=44100, time_len=10, threshold=5000):
    """

    Args:
        wav_path (str):
        fs (int): Hz
        time_len (int): sec
        threshold (int): Hz

    Returns:

    """

    a_path_list = glob.glob(Config.a_path)
    v_path_list = glob.glob(Config.v_path)

    a_path_list.sort()
    v_path_list.sort()

    a_path_list = a_path_list[-100:-10]
    v_path_list = v_path_list[-100:-10]

    a_power, a_freq = mean_spectrum(a_path_list)
    v_power, v_freq = mean_spectrum(v_path_list)

    sig = sf.read(wav_path)[0]
    center_idx = int(len(sig) / 2)
    sig = sig[center_idx - fs * time_len: center_idx + fs * time_len]
    power, freq = stft.power_spectrum(sig)
    th_idx = int(threshold/freq[1])
    power = power[:th_idx]

    if len(power) == len(a_power):
        a_rmse = np.sqrt(mean_squared_error(a_power, power))
        v_rmse = np.sqrt(mean_squared_error(v_power, power))

        if a_rmse < v_rmse:
            shutil.move(wav_path, Config.root_dir_path + "/A/")
        elif a_rmse > v_rmse:
            shutil.move(wav_path, Config.root_dir_path + "/V/")
        else:
            print("Indefinite")

    else:
        print('Recording time is short: ', wav_path)


def main():
    wav_path_list = glob.glob(Config.wav_dir_path)
    file_no = len(wav_path_list)

    for n, _path in enumerate(wav_path_list):
        process(_path)
        if n % 50 == 0:
            print(n, "/", file_no)


if __name__ == "__main__":
    main()
