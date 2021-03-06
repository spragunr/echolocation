"""
Utilizes a neural net to predict depth (specifically, distance
from wall) given audio data (inspired by bats' echolocation)

data by Nathan Sprague
developed by Nhung Hoang

May 2017
"""

import matplotlib.pyplot as plt
import numpy as np
import os.path

from keras.layers import Conv2D, Dense
from keras.layers.core import Flatten
from keras.models import load_model, Sequential
from random import shuffle
from scipy import io, signal
from sys import argv

def preprocess_data(filename):
    """
    @PURPOSE: Turn raw audio data into spectrograms and select target depths
    @PARAMS: filename - npz file of data 
    @RETURN: numpy array of spectrograms, numpy array of target depths
    """
    
    data = np.load(filename)
    audio = data['audio']
    depth = data['depth']

    # Depth vectors become single depth value 
    shrunk = depth[:, 230:250, 310:330] # square wall space directly in front of apparatus
    shrunk_reshaped = np.reshape(shrunk, (shrunk.shape[0],-1))
    target_set = np.max(shrunk_reshaped, axis=1)
    target_set = target_set[np.where(target_set!=0)]
    target_set = np.log(target_set) 

    # Turn audio data into spectrograms
    input_set = np.empty((audio.shape[0], 129, 11)) # (129, 11) shape of spectrograms
    for i in range(audio.shape[0]):
        freq, time, spectro = signal.spectrogram(audio[i,:])
	print spectro.shape
        input_set[i] = spectro
    input_set = np.log(input_set)/20
    
    # Shuffle the data 
    combined = zip(input_set, target_set)
    shuffle(combined)
    input_set, target_set = zip(*combined)
    input_set = np.asarray(input_set)
    target_set = np.asarray(target_set)

    return input_set, target_set

######################################################
    
def get_model(in_train, out_train):

    # Build neural net
    net = Sequential()
    INPUT_SHAPE = in_train.shape[1:]
    net.add(Conv2D(64, (5, 5), strides=(1,1), activation='relu', input_shape=INPUT_SHAPE))
    net.add(Flatten())
    net.add(Dense(450, activation='relu'))
    net.add(Dense(1, activation='linear'))
    net.compile(optimizer='adam', loss='mean_squared_error')
    net.fit(in_train, out_train, validation_split=0.2, epochs=50)
    net.save("depth_model.h5")
    return load_model("depth_model.h5")

######################################################

def run_nn(net, in_test, out_test, summary=False):
    """
    @PURPOSE: Train a neural network and use it to make predictions 
    @PARAMS: training_set - numpy array of spectrogram 
             target_set - numpy array of target depths for training set
             summary - boolean option to print neural network information
    @RETURN: mean squared error based on network performance using training and test sets 
    """
    
    loss = net.evaluate(in_test, out_test)
    scale_loss = np.exp(loss)

    predictions = net.predict(in_test)
    plot_data(np.exp(out_test), np.exp(predictions))
    
    if summary:
        print "\n"
        net.summary()
    return scale_loss 

######################################################

def plot_data(y_data, predictions):
    """
    @PURPOSE: display graph of expected depths vs. predicted depths for each data input
    @PARAMS: y_data - the true/expected depth values 
             prediction - the predicted depth values given by the neural network
    @RETURN: None
    """
    
    pts = 20 # number of data points to show
    indices = range(1, len(y_data)+1)
    plt.figure(1)
    plt.plot(indices[:pts], y_data[:pts], 'bs') 
    plt.plot(indices[:pts], predictions[:pts], 'g^')
    plt.title("True and Predicted Depth Values")
    plt.xlabel("data point")
    plt.ylabel("millimeters from wall")
    plt.legend(["true","predicted"])
    plt.show()
    
    plt.figure(2)
    plt.plot(y_data[:pts], predictions[:pts], 'ro')
    plt.title("True vs. Predicted Depth Values")
    plt.xlabel("true (mm)")
    plt.ylabel("predicted (mm)")
    plt.show()

######################################################
    
def main():
    if len(argv) != 3:
        print "\nusage: depth_detect.py training_data(npz_file) test_data(npz_file)\n"
        return

    # Get data sets
    in_train, out_train = preprocess_data(argv[1])
    in_train = np.reshape(in_train, (in_train.shape[0],129,11,1))
    in_test, out_test = preprocess_data(argv[2])
    in_test = np.reshape(in_test, (in_test.shape[0],129,11,1))
 
    if not os.path.isfile("depth_model.h5"):
	net = get_model(in_train, out_train)
    else:
	net = load_model("depth_model.h5")

    losses = []
    runs = 1
    for i in range(runs):
        if i == runs-1:
            loss = run_nn(net, in_test, out_test, summary=True)
        else:
            loss = run_nn(net, in_test, out_test)
        print "\n"
        losses.append(loss)

    print "\n"
    for j in range(runs):
        print "Run", j+1, "Loss:", losses[j]
    print "\nAverage Loss:", sum(losses)/runs
    print "\n","-"*30,"\n"," "*12,"DONE\n","-"*30
    
main()

######################################################
######################################################

def check_data():
    data = np.load(argv[1])
    depth = data['depth']
    print depth.shape	
    shrunk = depth[:, 230:250, 310:330] 
    shrunk_reshaped = np.reshape(shrunk, (shrunk.shape[0],-1))
    target_set = np.max(shrunk_reshaped, axis=1)
    target_set = target_set[np.where(target_set!=0)]
    target_set = np.log(target_set)
    plot_data(target_set, np.zeros_like(target_set))
#check_data()
