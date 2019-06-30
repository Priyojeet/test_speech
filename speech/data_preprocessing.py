from keras import utils as np_utils
import numpy as np
import librosa
import os





# get the mfcc of wav form
def wav2mfcc(file_path, max_pad_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)# res_type='kaiser_fast')
    print(wave, sr)
    wave = wave[::3]
    mfcc = np.mean(librosa.feature.mfcc(wave, sr=16000, n_mfcc = 40).T, axis=0)
    mfcc = np.pad(mfcc, pad_width=(0, 0), mode='constant')
    return mfcc

DATA_PATH = "data_new/"

# function to get the labels
def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, np_utils.to_categorical(label_indices)

# saving data to an numpy array
def save_data_to_array(path=DATA_PATH, max_pad_len=11):
    labels, _, _ = get_labels(path)

    for label in labels:
        # Init mfcc vectors
        mfcc_vectors = []

        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in wavfiles:
            mfcc = wav2mfcc(wavfile, max_pad_len=max_pad_len)
            mfcc_vectors.append(mfcc)
        np.save(label + '.npy', mfcc_vectors)


