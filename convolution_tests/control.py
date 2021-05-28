import numpy as np
import librosa
import scipy.signal as sps
import sounddevice as sd

sig, fs = librosa.core.load('filthy_48000.wav', sr=None)
sig2, fs2 = librosa.core.load('tunnel.wav', sr=fs)
win_len = 256
# for testing only
#sig2 = np.zeros(win_len)
#sig2[0] = 1.0

# another test
sig2 = np.zeros(win_len)
sig2[0:9] = np.array([0.00390625, 0.03125   , 0.109375  , 0.21875   , 0.2734375 , 0.21875   , 0.109375  , 0.03125   , 0.00390625])

outsig = sps.fftconvolve(sig, sig2[:win_len])
sd.play(outsig, fs)
sd.wait()
