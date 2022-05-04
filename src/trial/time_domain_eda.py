import glob
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import plotly.offline as offline
import plotly.graph_objects as go

from src.util.filt import Filter
from src.util.stft import Cepstrum


@dataclass
class Config:
    a_path: str = '../../data/raw/A/*.wav'
    v_path: str = '../../data/raw/V/*.wav'
    save_response_dir_path: str = "../../figure/"
    save_filtered_path: str = "../../figure/filtered.html"
    time_len: int = 10  # sec


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

path_list = glob.glob(Config.a_path)
path_list.sort()
data = read_wav(path_list[118])


_filt = Filter(data, Config.save_response_dir_path).low_pass(100, 200, 10)
_cep = Cepstrum(data)
# p, f, e = _cep.cepstrum()
# scatter1 = go.Scatter(x=t, y=c)
# fig = go.Figure(data=[scatter1])
# offline.plot(fig, filename="test.html", auto_open=True)


# p, f, e = _cep.envelope()
#
# scatter1 = go.Scatter(x=f, y=p, name="power")
# scatter2 = go.Scatter(x=f, y=e, name="env")
# fig = go.Figure(data=[scatter1, scatter2])
# offline.plot(fig, filename="test.html", auto_open=True)

# _filt = filt.Filter(data, Config.save_response_dir_path)
# filt_data = _filt.low_pass(30, 100, 30, 1, True)


p, f, e = _cep.envelope()
scatter1 = go.Scatter(x=f, y=p, name="power")
scatter2 = go.Scatter(x=f, y=e, name="power")


fig = go.Figure(data=[scatter1, scatter2])
offline.plot(fig, filename="test.html", auto_open=True)