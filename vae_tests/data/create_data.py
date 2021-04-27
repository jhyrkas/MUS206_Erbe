import json
import librosa
import numpy as np

def mtof(midi_note) :
    return 440.0 * pow(2.0, (midi_note - 69) / 12.0)

files = []
f0s = []

# test data
with open('nsynth-test/examples.json') as f:
  test_data = json.load(f)

for ex in test_data.keys() :
    files.append('nsynth-test/audio/' + ex + '.wav')
    for i in range(3) :
        f0s.append(mtof(test_data[ex]['pitch']))

# validation data
with open('nsynth-valid/examples.json') as f:
  valid_data = json.load(f)

for ex in valid_data.keys() :
    files.append('nsynth-valid/audio/' + ex + '.wav')
    for i in range(3) :
        f0s.append(mtof(valid_data[ex]['pitch']))

'''
# training data
with open('nsynth-train/examples.json') as f:
  valid_data = json.load(f)

for ex in valid_data.keys() :
    files.append('nsynth-train/audio/' + ex + '.wav')
    for i in range(3) :
        f0s.append(mtof(valid_data[ex]['pitch']))
'''

num_examples = len(files) * 3

nfft = 2048
freq_size = nfft // 2 + 1
timbre_data = np.zeros((freq_size, num_examples))
examples_processed = 0

for i in range(len(files)) :
    ex = files[i]
    sig, fs = librosa.core.load(ex, sr = None)
    spect = np.abs(librosa.stft(sig, n_fft = nfft))
    # spect of top ten loudest frames to try to avoid silence
    spect_sum = np.sum(spect, axis = 0).flatten()
    top_three = spect_sum.argsort()[-3:][::-1]
    for j in range(len(top_three)) :
        c = top_three[j]
        timbre_data[:,(3*i)+j] = spect[:, c]
        examples_processed += 1

assert(examples_processed == num_examples)

f0s = np.array(f0s)
np.random.seed(12345)
sorting_array = np.arange(num_examples)
np.random.shuffle(sorting_array)

np.save('timbre_data.npy', timbre_data[:,sorting_array])
np.save('pitch_data.npy', f0s[sorting_array])
