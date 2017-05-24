"""
Utilizes a neural net to predict depth (specifically, distance
from wall) given audio data (inspired by bats' echolocation)

data by Nathan Sprague
developed by Nhung Hoang

May 2017
"""
import matplotlib.pyplot as plt
import numpy as np
'''
from keras.layers import Conv2D, Dense
from keras.layers.core import Flatten
from keras.models import Sequential
from random import shuffle'''
from scipy import io, signal
from sys import argv

def preprocess_data():
    data = np.load(argv[1])
    audio = data['audio']
    depth = data['depth']

    shrunk = depth[:, 230:250, 310:330] # square wall space directly in front of mic
    shrunk_reshaped = np.reshape(shrunk, (1159,-1))
    target_set = np.max(shrunk_reshaped, axis=1)
    target_set = np.log(target_set) 

    # Turn audio data into spectrograms
    training_set = np.empty((1159, 129, 11)) # (129, 11) shape of 1159 spectrograms
    for i in range(audio.shape[0]):
        freq, time, spectro = signal.spectrogram(audio[i,:])
        training_set[i] = spectro
    training_set = np.log(training_set)/20
    return training_set, target_set

######################################################
    
def run_nn(training_set, target_set, summary=False):
 
    # Shuffle the data 
    combined = zip(training_set, target_set)
    shuffle(combined)
    training_set, target_set = zip(*combined)
    training_set = np.asarray(training_set)
    target_set = np.asarray(target_set)
    
    '''
    # Show plots
    print "MIN:",np.min(training_set)
    print "MAX:",np.max(training_set)
    plt.hist(training_set.flatten(), 1000)
    plt.show()
    
    plt.pcolormesh(time, freq, spectro)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    '''

    # Using these sets for now
    in_train = training_set[:928]
    in_train = np.reshape(in_train, (928,129,11,1))
    out_train = target_set[:928]
    in_test = training_set[928:]
    in_test = np.reshape(in_test, (231,129,11,1))
    out_test = target_set[928:]

    # Build neural net
    net = Sequential()
    INPUT_SHAPE = in_train.shape[1:]
    net.add(Conv2D(64, (5, 5), strides=(1,1), activation='relu', input_shape=INPUT_SHAPE))
    net.add(Flatten())
    net.add(Dense(450, activation='relu'))
    net.add(Dense(1, activation='linear'))
    net.compile(optimizer='adam', loss='mean_squared_error')
    net.fit(in_train, out_train, validation_split=0.2, epochs=15)
    loss = net.evaluate(in_test, out_test)
    scale_loss = np.exp(loss)

    predictions = net.predict(in_test)
    plot_data(out_test, predictions)
    
    if summary:
        print "\n"
        net.summary()
    return scale_loss 

######################################################

def plot_data(y_data, predictions):
    pts = 20 # number of data points to show
    indices = range(1, len(y_data)+1)
    plt.plot(indices[:pts], y_data[:pts], 'bs') 
    plt.plot(indices[:pts], predictions[:pts], 'g^')
    plt.xlabel("data point")
    plt.ylabel("millimeters from wall")
    plt.legend(["true","predicted"])
    plt.show()
    
######################################################
    
def main():
    if len(argv) != 2:
        print "\nusage: depth_detect.py npz_file\n"
        return

    training_set, target_set = preprocess_data()
    
    losses = []
    runs = 1
    for i in range(runs):
        if i == runs-1:
            loss = run_nn(training_set, target_set, summary=True)
        else:
            loss = run_nn(training_set, target_set)
        print "\n"
        losses.append(loss)

    print "\n"
    for j in range(runs):
        print "Run", j+1, "Loss:", losses[j]
    print "\nAverage Loss:", sum(losses)/runs
    print "\n","-"*30,"\n"," "*12,"DONE\n","-"*30
    
#main()

def check_data():
    data = np.load(argv[1])
    depth = data['depth']

    print depth.shape	

    shrunk = depth[35:175, 230:250, 310:330] # square wall space directly in front of mic
    shrunk_reshaped = np.reshape(shrunk, (shrunk.shape[0],-1))
    target_set = np.max(shrunk_reshaped, axis=1)
    binary = np.zeros_like(target_set)
    binary[target_set==0] = 1
    print binary
    print np.where(binary==1)
    target_set = np.log(target_set) 
    plot_data(target_set, np.zeros_like(target_set))

check_data()
