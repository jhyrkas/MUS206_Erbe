import librosa
import math
import numpy as np
import sounddevice as sd
import torch
import sys

from vae_cqt import vae_cqt
from vae_stft import vae_stft

# will probably add some functions to this so let's make a main
if __name__ == '__main__' :
    use_stft = int(sys.argv[1]) == 0
    gain = 0.75

    if use_stft :
        vae = vae_stft()
        vae.load_state_dict(torch.load('vae_stft_model_params.pytorch'))
        vae.eval()
        timbre_data = np.load('data/timbre_data.npy').T
        good_rows = np.max(timbre_data, axis=1) > 0
        timbre_data = timbre_data[good_rows,:]
        timbre_data = timbre_data / np.max(timbre_data, axis=1).reshape(timbre_data.shape[0], 1)
        timbre_data = torch.from_numpy(timbre_data).float()
        pitch_data = np.load('data/pitch_data.npy')
        pitch_data = pitch_data[good_rows]
        #pitch_data = torch.from_numpy(np.resize(pitch_data, ((len(pitch_data), 1)))).float()
        fs = 16000
        hop_length = 512
        length = 3 # seconds
        n_reps = length * int(fs / hop_length)
        data_size = 1025
    else :
        vae = vae_cqt()
        vae.load_state_dict(torch.load('vae_cqt_model_params.pytorch'))
        vae.eval()
        timbre_data = np.load('data/timbre_data_cqt.npy').T
        timbre_data = timbre_data / np.max(timbre_data, axis=1).reshape(timbre_data.shape[0], 1)
        timbre_data = torch.from_numpy(timbre_data).float()
        pitch_data = np.load('data/pitch_data_cqt.npy')
        #pitch_data = torch.from_numpy(np.resize(pitch_data, ((len(pitch_data), 1)))).float()
        fs = 16000
        length = 3 # seconds
        hop_length = 128
        n_reps = length * int(fs / hop_length)
        n_octaves = 7
        bins_per_octave = 36
        data_size = 252

    #examples = np.random.choice(np.arange(timbre_data.shape[0]), size=10, replace=False)
    low_pitches = np.where(np.logical_and(pitch_data < 220, pitch_data > 60))[0] # tuple for some reason?
    examples = np.random.choice(low_pitches, size=10, replace=False)
    for i in range(10) :
        X = timbre_data[examples[i], :].reshape(1, data_size)
        f0 = pitch_data[examples[i]]
        mu, logvar = vae.encode(X)
        z = vae.reparam_trick(mu, logvar)
        X_hat = vae.decode(z).detach()
        #X_hat = vae.decode(mu, f0)
        #print(X_hat)

        if use_stft :
            x = librosa.griffinlim(np.repeat(X.numpy().reshape(data_size,1), n_reps, axis=1))
            x_hat = librosa.griffinlim(np.repeat(X_hat.numpy().reshape(data_size,1), n_reps, axis=1))
        else :
            C = np.repeat(X.numpy().reshape(data_size,1), n_reps, axis=1)
            test_sig = librosa.griffinlim_cqt(C, sr=fs, hop_length = hop_length, bins_per_octave = bins_per_octave, n_iter=1)
            x = librosa.griffinlim_cqt(np.repeat(X.numpy().reshape(data_size,1), n_reps, axis=1), sr=fs, hop_length = hop_length, bins_per_octave = bins_per_octave)
            x_hat = librosa.griffinlim_cqt(np.repeat(X_hat.detach().numpy().reshape(data_size,1), n_reps, axis=1), sr=fs, hop_length = hop_length, bins_per_octave = bins_per_octave)

        x = gain * (x / np.max(np.abs(x)))
        x_hat = gain * (x_hat / np.max(np.abs(x_hat)))
        f0_frames, voiced_, _ = librosa.pyin(x, librosa.note_to_hz('C2'), librosa.note_to_hz('C7'), sr=fs)
        f0_hat_frames, voiced_hat, _ = librosa.pyin(x_hat, librosa.note_to_hz('C2'), librosa.note_to_hz('C7'), sr=fs)
        f0_ = np.mean(f0_frames[voiced_]) if np.sum(voiced_) > 0 else 0
        f0_hat = np.mean(f0_hat_frames[voiced_hat]) if np.sum(voiced_hat) > 0 else 0
        print(str(f0) + '\t' + str(f0_) + '\t' + str(f0_hat))
        new_fs = 48000
        new_x_hat = librosa.resample(x_hat, fs, new_fs)
        new_x_hat = new_x_hat / np.max(np.abs(new_x_hat))
        cycle_samps = int(round(new_fs/f0_hat))
        start_index = new_fs//2 # avoid silence at beginning? 
        looping = True
        while looping and start_index < len(new_x_hat):
            if math.isclose(new_x_hat[start_index], 0.0, abs_tol=0.001) :
                looping = False
            else :
                start_index += 1

        if start_index + cycle_samps <= len(new_x_hat) :
            sd.play(x_hat, fs)
            sd.wait()
            sig_rep = np.tile(new_x_hat[start_index:start_index+cycle_samps], int(round((3*new_fs)/cycle_samps)))
            sd.play(sig_rep * 0.8, new_fs)
            sd.wait()
        else :
            print(new_x_hat[0:cycle_samps])
        
        #sd.play(x, fs)
        #sd.wait()
        #sd.play(x_hat, fs)
        #sd.wait()
