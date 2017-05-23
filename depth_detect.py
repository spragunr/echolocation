"""
Utilizes a neural net to predict depth (specifically, distance
from wall) given audio data (inspired by bats' echolocation)

data by Nathan Sprague
developed by Nhung Hoang

May 18, 2017
"""
from keras.layers import Conv2D, Dense
from keras.layers.core import Flatten
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
from scipy import io, signal #io.wavfile, signal.spectrogram
from sys import argv

def main():
    if len(argv) != 2:
        print "\nusage: depth_detect.py npz_file\n"
        return

    data = np.load(argv[1])
    audio = data['audio']
    depth = data['depth']

    shrunk = depth[:, 230:250, 310:330]
    shrunk_reshaped = np.reshape(shrunk, (1159,-1))
    target_set = np.max(shrunk_reshaped,axis=1)

    '''
    fs = 10e3
    N = 1e5
    amp = 2 * np.sqrt(2)
    noise_power = 0.01 * fs / 2
    time = np.arange(N) / float(fs)
    mod = 500*np.cos(2*np.pi*0.25*time)
    carrier = amp * np.sin(2*np.pi*3e3*time + mod)
    noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
    noise *= np.exp(-time/5)
    x = carrier + noise
    '''

    # Turn audio data into spectrograms
    training_set = np.empty((1159, 129, 11)) # shape of 1159 spectrograms
    for i in range(audio.shape[0]):
        freq, time, spectro = signal.spectrogram(audio[i,:])
        training_set[i] = spectro

    print "MINS"
    print np.min(training_set)
    print np.max(training_set)
    plt.hist(np.log(training_set.flatten()), 1000)
    plt.show()
    print
    '''
    plt.pcolormesh(time, freq, spectro)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    '''

    training_set = np.log(training_set)/20.0
    target_set = np.log(target_set)

    # Using these sets for now
    in_train = training_set[:928]
    out_train = target_set[:928]
    in_test = training_set[928:]
    out_test = target_set[928:]

    # Build neural net
    net = Sequential()
    INPUT_SHAPE = (in_train.shape[1], in_train.shape[2],1)
    net.add(Conv2D(64, (5, 5), strides=(1,1), activation='relu', input_shape=INPUT_SHAPE))
    net.add(Flatten())
    net.add(Dense(1, activation='relu'))
    net.compile(optimizer='adam', loss='mean_squared_error')
    print in_train.shape
    in_train = np.reshape(in_train, (928, 129, 11, 1))
    net.fit(in_train, out_train, validation_split=0.2)

    loss = net.evaluate(in_test, out_test)
    print "\nloss: ", loss

    print "\n","-"*30,"\n"," "*12,"DONE\n","-"*30

main()
