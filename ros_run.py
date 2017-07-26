#!/usr/bin/env python

"""Ros node for for reconstructing depth using a pre-trained network.
This is a hacked-together proof-of-concept.

"""
import time
import subprocess
import h5py
import threading
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters

from sound_play.libsoundplay import SoundClient

from pyaudio_utils import AudioPlayer, AudioRecorder

from keras.models import load_model
import align_audio
import tensorflow as tf
from keras.backend import floatx

class Recorder(object):

    def __init__(self):
        rospy.init_node('ros_record')
        self.parse_command_line()
        model_file = 'stereo/model_100k3_raw_short.h5'
        self.model = load_model(model_file,
                                custom_objects={'adjusted_mse':adjusted_mse})
        self.parse_command_line()

        self.image_pub = rospy.Publisher("predicted_depth",Image)

        subprocess.call(["amixer", "-D", "pulse", "sset",
                         "Master", "{}%".format(self.volume)])
        subprocess.call(["amixer", "-D", "pulse", "sset",
                         "Capture", "{}%".format(self.mic_level)])

        self.bridge = CvBridge()

        self.audio_player = AudioPlayer(self.chirp_file)
        self.audio_recorder = AudioRecorder(channels=self.channels)

        rate = rospy.Rate(self.rate)

        line = None
        im = None
        # MAIN LOOP
        while not rospy.is_shutdown():

            # Play and record audio
            self.audio_player.play()
            rospy.sleep(self.delay) # hack.it takes the sound a while to play...
            self.audio_recorder.record(self.record_duration)

            audio = self.record()

            # Align and shape the audio for the network
            aligned = align_audio.align_clip(audio)
            aligned = aligned[0:3328,:]
            aligned = np.append(aligned[:,0], aligned[:,1])
            aligned = aligned / 32000.
            aligned = np.reshape(aligned, (1,aligned.size, 1))

            # Get the depth prediction from the network
            predictions = self.model.predict(aligned, batch_size=1)
            predictions = np.exp(np.reshape(predictions, (12,16))) - 1
            
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(predictions))

            # Use matplotlib to show the audio and predicted depth
            plt.ion()
            plt.show()

            axes0 = plt.subplot(2,1,1)
            axes1 = plt.subplot(2,1,2)
            if line is not None:
                line.remove()
                im.remove()
            line, = axes0.plot(aligned[0,:,0])
            im = axes1.imshow(predictions,  clim=(300, 7000),
                              interpolation='none')
            plt.draw()
            plt.pause(1e-17)

                                 
            rate.sleep()



        self.audio_player.shutdown()
        self.audio_recorder.shutdown()


    def record(self):
        self.audio_recorder.record(self.record_duration)
        while not self.audio_recorder.done_recording():
            rospy.sleep(.005)
                
        audio = self.audio_recorder.get_data()[1]

        # Reshape mono to be consistent with stereo
        if (len(audio.shape) == 1):
            audio = audio.reshape((-1, 1))
        return audio
        
    def parse_command_line(self):

        parser = argparse.ArgumentParser(
            description='Sonar/image/depth data collection tool')

        parser.add_argument('--num-channels', type=int,
                            dest='channels',
                            metavar="NUM_CHANNELS",default=2,
                            help='number of audio channels to record')
        
        parser.add_argument('--rate', type=int, metavar="RATE",
                            default=10, help='rate to record chirps')

        parser.add_argument('--duration', type=float, metavar="DURATION",
                            dest='record_duration',
                            default=.11, help='duration of audio recordings')
        
        parser.add_argument('--delay', type=float, metavar="DELAY",
                            default=.0, help=('time in seconds to wait' +
                                               'start of playback and record'))
        
        parser.add_argument('--volume', type=int, metavar="VOLUME",
                            default=75, help='volume (0-100)')

        parser.add_argument('--mic-level', type=int, metavar="MIC_LEVEL",
                            dest='mic_level',
                            default=100, help='mic_level (0-100)')

        parser.add_argument('-c', '--chirp-file', type=str, metavar="CHIRP_FILE",
                            default='data/16000to8000.02s.wav',
                            help='Location of .wav file.')


        parser.parse_args(namespace=self)


    def init_data_sets(self):
        self.h5_file = h5py.File(self.out, 'w')
        test_audio = self.record()
        self.audio_set = self.h5_file.create_dataset('audio',
                                                     (1, test_audio.shape[0], self.channels),
                                                     maxshape=(None,
                                                               test_audio.shape[0],
                                                               self.channels),
                                                     dtype=np.int16)


        depth_shape = self.latest_depth.shape
        self.depth_set = self.h5_file.create_dataset('depth', (10,
                                                          depth_shape[0],
                                                          depth_shape[1]),
                                                maxshape=(None,
                                                          depth_shape[0],
                                                          depth_shape[1]),
                                                dtype=self.latest_depth.dtype)
        if self.record_rgb:
            rgb_shape = self.latest_rgb.shape
            self.rgb_set = self.h5_file.create_dataset('rgb', (10,
                                                               rgb_shape[0],
                                                               rgb_shape[1],
                                                               rgb_shape[2]),
                                                       maxshape=(None,
                                                                 rgb_shape[0],
                                                                 rgb_shape[1],
                                                                 rgb_shape[2]),
                                                       dtype=self.latest_rgb.dtype)
        self.time_set = self.h5_file.create_dataset('time', (1,),
                                                     maxshape=(None,),
                                                    compression="lzf",
                                                     dtype=np.float64)

    def close_file(self, num_recorded):
        self.audio_set.resize(tuple([num_recorded] +
                                    list(self.audio_set.shape[1:])))
                                                          
        self.depth_set.resize(tuple([num_recorded] +
                                    list(self.depth_set.shape[1:])))
        if self.record_rgb:
            self.rgb_set.resize(tuple([num_recorded] +
                                      list(self.rgb_set.shape[1:])))
        self.time_set.resize((num_recorded,))

        self.h5_file.close()

    def h5_append(self, dset, index, item):
        if index == dset.shape[0]:
            dset.resize(tuple([index*2] + list(dset.shape[1:])))
        dset[index, ...] = item
            

    def depth_callback(self, depth_image):
        self.latest_depth = self.bridge.imgmsg_to_cv2(depth_image)
        
    def depth_rgb_callback(self, depth_image, rgb_image):
        self.lock.acquire()
        self.latest_depth = self.bridge.imgmsg_to_cv2(depth_image)
        self.latest_rgb = self.bridge.imgmsg_to_cv2(rgb_image,
                                                    "rgb8")
        self.lock.release()

def adjusted_mse(y_true, y_pred):
    zero = tf.constant(0, dtype=floatx())
    ok_entries = tf.not_equal(y_true, zero)
    safe_targets = tf.where(ok_entries, y_true, y_pred)
    sqr = tf.square(y_pred - safe_targets)
    valid = tf.cast(ok_entries, floatx())
    num_ok = tf.reduce_sum(valid, axis=-1) # count OK entries
    num_ok = tf.maximum(num_ok, tf.ones_like(num_ok)) # avoid divide by zero
    return tf.reduce_sum(sqr, axis=-1) / num_ok

if __name__ == "__main__":
    Recorder()


