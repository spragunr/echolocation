#!/usr/bin/env python

"""Ros node for playing audio chirps and recording the returns along
with data from a depth camera.

This requires that the depth camera has been started (in a separate
terminal.):

source ./ros_config_account.sh

roslaunch openni2_launch openni2.launch

"""
import time
import subprocess
import wave
import threading
import argparse

import h5py
import scipy.io.wavfile
import numpy as np

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters

import sounddevice as sd

class Recorder(object):

    def __init__(self):
        rospy.init_node('ros_record')

        self.parse_command_line()

        subprocess.call(["amixer", "-D", "pulse", "sset",
                         "Master", "{}%".format(self.volume)])
        subprocess.call(["amixer", "-D", "pulse", "sset",
                         "Capture", "{}%".format(self.mic_level)])

        self.lock = threading.Lock()

        self.bridge = CvBridge()
        self.latest_depth = None
        self.latest_rgb = None

        if not self.record_rgb:
            rospy.Subscriber('/camera/depth/image_raw', Image,
                             self.depth_callback)

        else:
            ds = message_filters.Subscriber('/camera/depth/image_raw', Image)
            rgbs = message_filters.Subscriber('/camera/rgb/image_raw', Image)
            ts = message_filters.ApproximateTimeSynchronizer([ds, rgbs], 10, .03)
            ts.registerCallback(self.depth_rgb_callback)

        self.init_audio()

        while self.latest_depth is None and not rospy.is_shutdown():
            rospy.loginfo("WAITING FOR CAMERA DATA.")
            rospy.sleep(.1)

        while self.latest_recording is None and not rospy.is_shutdown():
            rospy.loginfo("WAITING FOR AUDIO DATA.")
            rospy.sleep(.1)

        self.init_data_sets()

        rate = rospy.Rate(60)
        index = 0

        # MAIN LOOP
        while not rospy.is_shutdown():
            
            self.lock.acquire()
            if self.last_storage_time != self.latest_recording_time:
                # grab latest data
                audio = self.latest_recording
                depth_image = self.latest_depth
                if self.record_rgb:
                    rgb_image = self.latest_rgb
                self.last_storage_time = self.latest_recording_time
                self.lock.release()
                
                # store on disk
                self.h5_append(self.depth_set, index, depth_image)
                self.h5_append(self.audio_set, index, self.latest_recording)
                if self.record_rgb:
                    self.h5_append(self.rgb_set, index, rgb_image)

                self.h5_append(self.time_set, index, self.last_storage_time)
                index += 1
            else:
                self.lock.release()

            if not self.stream.active:
                print "restarting stream..."
                self.stream.close()
                self.init_audio()
                
            rate.sleep()

        # MAIN LOOP COMPLETE...
        self.stream.stop()
        self.stream.close()
        self.close_file(index - 1)

    def reset_audio_counters(self):
        self.last_storage_time = -1
        self.chirp_index = 0 # where are we in the current chirp (used
                             # by callback)
        self.cur_block = 0 # used by callblack to count blocks in the
                           # current chirp
        self.record_index = 0
        self.latest_recording_time = -1;

    def init_audio(self):
        # start audio, and wait for first audio sample
        self.reset_audio_counters()
        self.chirp_file = wave.open(self.chirp_file_name, 'rb')
        rate = self.chirp_file.getframerate()
        self.blocksize = 128
        self.record_delay = .085 # at least (will be somewhat more...)
        self.record_delay_blocks = np.ceil(self.record_delay /
                                           (self.blocksize / float(rate)))
        self.record_blocks = np.ceil(self.record_duration /
                                     (self.blocksize / float(rate)))
        self.chirp_delay = 0
        self.chirp_delay_blocks = np.ceil(self.chirp_delay /
                                          (self.blocksize / float(rate)))
        self.total_blocks = (self.record_blocks + self.record_delay_blocks +
                             self.chirp_delay_blocks)
        print "TOTAL_BLOCKS", self.total_blocks
        self.latest_recording = None
        self.current_recording = np.zeros((int(self.record_blocks *
                                               self.blocksize),
                                           self.channels),
                                          dtype='int16')

        f = open(self.chirp_file_name, 'rb')
        self.chirp_data = scipy.io.wavfile.read(f)[1]
        if len(self.chirp_data.shape) == 1:
            self.chirp_data = self.chirp_data.reshape((-1, 1))


        self.stream = sd.Stream(device=(None, None),
                           samplerate=self.chirp_file.getframerate(),
                           blocksize=self.blocksize, dtype='int16',
                           channels=(self.channels,
                                     self.chirp_file.getnchannels()),
                           callback=self.audio_callback)


        self.stream.start()

        
    def audio_callback(self, indata, outdata, frames, callback_time, status):
        if status:
            print status
            self.stream.close()
            return
        
        # HANDLE PLAYBACK
        if self.chirp_index != -1:
            if self.chirp_index + frames <= self.chirp_data.shape[0]:
                outdata[:,...] = self.chirp_data[self.chirp_index:self.chirp_index+frames,...]
                self.chirp_index += frames
            else:
                left = self.chirp_data.shape[0] - self.chirp_index
                outdata[0:left, ...] = self.chirp_data[self.chirp_index::, ...]
                outdata[left::,...] = np.zeros((frames - left, self.chirp_data.shape[1]))
                self.chirp_index = -1
        else:
            outdata[:] =  np.zeros(outdata.shape)


        # HANDLE RECORDING
        # If we need to store the current audio block...
        if (self.cur_block >= self.record_delay_blocks and
            self.cur_block < self.record_delay_blocks + self.record_blocks):
            self.current_recording[self.record_index:self.record_index+frames,...] = indata
            self.record_index += frames
            
        # If we just stored the last block...
        if (self.cur_block == self.record_delay_blocks + self.record_blocks - 1):
            self.lock.acquire()
            
            self.latest_recording = np.array(self.current_recording)
            self.latest_recording_time = time.time()
            self.lock.release()

        if self.cur_block >= self.total_blocks:
            self.chirp_index = 0
            self.record_index = 0
            self.cur_block = -1

        self.cur_block += 1
        

            
    def parse_command_line(self):

        parser = argparse.ArgumentParser(
            description='Sonar/image/depth data collection tool')

        parser.add_argument('out', type=str, metavar="OUT",
                            help='output file name')
        
        parser.add_argument('--num-channels', type=int,
                            dest='channels',
                            metavar="NUM_CHANNELS",default=2,
                            help='number of audio channels to record')
        
        parser.add_argument('--rate', type=int, metavar="RATE",
                            default=6, help='rate to record chirps')
        
        parser.add_argument('--duration', type=float, metavar="DURATION",
                            dest='record_duration',
                            default=.06, help='duration of audio recordings')

        parser.add_argument('--volume', type=int, metavar="VOLUME",
                            default=75, help='volume (0-100), default 75')
        parser.add_argument('--mic-level', type=int, metavar="MIC_LEVEL",
                            dest='mic_level',
                            default=100, help='mic_level (0-100)')

        parser.add_argument('-c', '--chirp-file', type=str,
                            dest='chirp_file_name',                            
                            metavar="CHIRP_FILE",
                            default='data/16000to8000.02s.wav',
                            help='Location of .wav file.')


        parser.add_argument('--no-rgb',  dest='record_rgb',action='store_false')
    

        parser.parse_args(namespace=self)


    def init_data_sets(self):
        self.h5_file = h5py.File(self.out, 'w')
        test_audio = self.latest_recording
        self.audio_set = self.h5_file.create_dataset('audio', (1,
                                                               test_audio.shape[0],
                                                               self.channels),
                                                     maxshape=(None,
                                                               test_audio.shape[0],
                                                               self.channels),
#                                                     compression="lzf",
                                                     dtype=np.int16)


        depth_shape = self.latest_depth.shape
        self.depth_set = self.h5_file.create_dataset('depth', (10,
                                                          depth_shape[0],
                                                          depth_shape[1]),
                                                maxshape=(None,
                                                          depth_shape[0],
                                                          depth_shape[1]),
#                                                     compression="lzf",
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
#                                                       compression="lzf",
                                                       dtype=self.latest_rgb.dtype)
        self.time_set = self.h5_file.create_dataset('time', (1,),
                                                     maxshape=(None,),
#                                                    compression="lzf",
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



if __name__ == "__main__":
    Recorder()


