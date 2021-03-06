#!/usr/bin/env python

"""Ros node for playing audio chirps and recording the returns along
with data from a depth camera.

This requires that the depth camera and the sound play node have been
started (in a separate terminal.):

source ./ros_config_account.sh

roslaunch openni2_launch openni2.launch
roslaunch sound_play soundplay_node.launch

"""
import time
import subprocess
import h5py
import threading
import sys
import argparse
import numpy as np

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters

from sound_play.libsoundplay import SoundClient

from pyaudio_utils import AudioPlayer, AudioRecorder

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

        #self.soundhandle = SoundClient(blocking=False)
        self.audio_player = AudioPlayer(self.chirp_file)
        self.audio_recorder = AudioRecorder(channels=self.channels)

        if not self.record_rgb:
            rospy.Subscriber('/camera/depth/image_raw', Image,
                             self.depth_callback)

        else:
            ds = message_filters.Subscriber('/camera/depth/image_raw', Image)
            rgbs = message_filters.Subscriber('/camera/rgb/image_raw', Image)
            ts = message_filters.ApproximateTimeSynchronizer([ds, rgbs], 10, .03)
            ts.registerCallback(self.depth_rgb_callback)

        while self.latest_depth is None and not rospy.is_shutdown():
            rospy.loginfo("WAITING FOR CAMERA DATA.")
            rospy.sleep(.1)

        self.init_data_sets()

        rate = rospy.Rate(self.rate)
        index = 0

        # MAIN LOOP
        while not rospy.is_shutdown() and index <= self.number:

            callback_time = time.time()

            # Grab image data
            if self.record_rgb:
                self.lock.acquire()
                depth_image = self.latest_depth
                rgb_image = self.latest_rgb
                self.lock.release()
            else:
                depth_image = self.latest_depth

            # Play and record audio
            self.audio_player.play()
            #self.soundhandle.playWave(self.chirp_file)
            rospy.sleep(.04) # hack.  it takes the sound a while to play...
            self.audio_recorder.record(self.record_duration)
            #self.soundhandle.playWave(self.chirp_file)

            audio = self.record()

            # Store to disk
            self.h5_append(self.time_set, index, callback_time)
            self.h5_append(self.audio_set, index, audio)
            self.h5_append(self.depth_set, index, depth_image)
            if self.record_rgb:
                self.h5_append(self.rgb_set, index, rgb_image)

            index += 1

            rate.sleep()


        self.close_file(index-1)
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

        parser.add_argument('out', type=str, metavar="OUT",
                            help='output file name')
        
        parser.add_argument('--num-channels', type=int,
                            dest='channels',
                            metavar="NUM_CHANNELS",default=2,
                            help='number of audio channels to record')
        
        parser.add_argument('--rate', type=int, metavar="RATE",
                            default=10, help='rate to record chirps')

        parser.add_argument('--number', type=int, metavar="NUMBER",
                            default=5000, help='number of chirps to record')
        
        parser.add_argument('--duration', type=float, metavar="DURATION",
                            dest='record_duration',
                            default=.08, help='duration of audio recordings')

        parser.add_argument('--volume', type=int, metavar="VOLUME",
                            default=75, help='volume (0-100)')
        parser.add_argument('--mic-level', type=int, metavar="MIC_LEVEL",
                            dest='mic_level',
                            default=100, help='mic_level (0-100)')

        parser.add_argument('-c', '--chirp-file', type=str, metavar="CHIRP_FILE",
                            default='data/16000to8000.02s.wav',
                            help='Location of .wav file.')


        parser.add_argument('--no-rgb',  dest='record_rgb',action='store_false')
    

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



if __name__ == "__main__":
    Recorder()


