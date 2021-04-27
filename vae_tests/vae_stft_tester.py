import librosa
import numpy as np
import sounddevice as sd
import torch
import sys

from vae_stft import vae_stft

# will probably add some functions to this so let's make a main
if __name__ == '__main__' :
    fs = 16000
    hop_length = 512
    length = 3 # seconds
    n_reps = length * int(fs / hop_length)
    modulate = len(sys.argv) > 1 and int(sys.argv[1]) == 1
    gain = 0.75

    vae = vae_stft()
    vae.load_state_dict(torch.load('vae_stft_model_params.pytorch'))
    vae.eval()

    timbre_data = np.load('data/timbre_data.npy').T
    timbre_data = timbre_data / np.max(timbre_data, axis=1).reshape(timbre_data.shape[0], 1)
    timbre_data = torch.from_numpy(timbre_data).float()
    pitch_data = np.load('data/pitch_data.npy')
    pitch_data = torch.from_numpy(np.resize(pitch_data, ((len(pitch_data), 1)))).float()

    examples = np.random.choice(np.arange(timbre_data.shape[0]), size=10, replace=False)
    for i in range(10) :
        X = timbre_data[examples[i], :].reshape(1, 1025)
        f0 = pitch_data[examples[i], :].reshape(1, 1)
        mu, logvar = vae.encode(X)
        z = vae.reparam_trick(mu, logvar)
        X_hat = vae.decode(z).detach()
        #X_hat = vae.decode(mu, f0)
        #print(X_hat)

        x = librosa.griffinlim(np.repeat(X.numpy().reshape(1025,1), n_reps, axis=1))
        x_hat = librosa.griffinlim(np.repeat(X_hat.numpy().reshape(1025,1), n_reps, axis=1))

        x = gain * (x / np.max(np.abs(x)))
        x_hat = gain * (x_hat / np.max(np.abs(x_hat)))

        zeroish_mask = X.numpy() < 0.0005
        zeroish_xhat = X_hat.numpy() < 0.0005
        mse = np.mean(np.square(X.numpy()[zeroish_mask] - X_hat.numpy()[zeroish_mask]))
        print ((np.sum(zeroish_mask) / zeroish_mask.shape[1], mse))

        sd.play(x, fs)
        sd.wait()
        sd.play(x_hat, fs)
        sd.wait()

        if modulate :
            z_mod = np.repeat(z.detach().numpy(), n_reps, axis=0)
            lfo = 1.0 * np.sin(2*np.pi*np.linspace(0, 1, n_reps)).reshape(n_reps, 1)
            z_mod = z_mod + lfo
            X_hat_mod = np.zeros((1025, n_reps))
            for j in range(n_reps) :
                next_z = torch.from_numpy(z_mod[j,:]).float()
                X_hat_mod[:,j] = vae.decode(next_z).detach().numpy()
            x_hat_mod = librosa.griffinlim(X_hat_mod)
            x_hat_mod = gain * (x_hat_mod / np.max(np.abs(x_hat_mod)))
            sd.play(x_hat_mod, fs)
            sd.wait()
