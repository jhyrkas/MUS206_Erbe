import math
import numpy as np
import librosa
import scipy.signal as sps
from scipy.fft import fft, ifft
import sounddevice as sd

win_len = 256
hop_len = win_len // 4
sig, fs = librosa.core.load('filthy_48000.wav', sr=None)
pad_len = len(sig) + win_len - 1
pad_len = pad_len + (win_len - pad_len % win_len)
padded_sig = np.zeros(pad_len)
padded_sig[0:len(sig)] = sig # is this right?
outsig = np.zeros(pad_len) # is this right?

sig2, fs2 = librosa.core.load('tunnel.wav', sr=fs)

# for basic testing
sig2 = np.zeros(win_len)
sig2[0] = 1.0

# another test
sig2 = np.zeros(win_len)
sig2[0:9] = np.array([0.00390625, 0.03125   , 0.109375  , 0.21875   , 0.2734375 , 0.21875   , 0.109375  , 0.03125   , 0.00390625])

filter_td = sig2[0:win_len]
window = sps.windows.hann(win_len)
filter_fd = fft(filter_td)

for i in range(len(sig) // hop_len) :
    start = i*hop_len
    end = start + win_len
    tmp = padded_sig[start:end] * window
    tmp_fd = fft(tmp)
    convolved = ifft(tmp_fd * filter_fd)
    outsig[start:end] += convolved.real

#outsig = sps.fftconvolve(sig, sig2)
sd.play(outsig, fs)
sd.wait()
