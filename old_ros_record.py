#!/usr/bin/env python

"""Ros node for playing audio chirps and recording the returns along
with data from a depth camera.

This requires that the depth camera and sound_play nodes have been
started (in separate terminals.):

source ./ros_config_account.sh

roslaunch openni2_launch openni2.launch
roslaunch sound_play soundplay_node.launch

"""
#import h5py
import sys
import numpy as np

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from sound_play.libsoundplay import SoundClient

from pyaudio_utils import AudioPlayer, AudioRecorder

CHIRP_FILE = '/home/hoangnt/echolocation/data/16000to8000.02s.wav'

class Recorder(object):

    CHIRP_RATE = 6 # in hz
    RECORD_DURATION = .06 # in seconds
    CHANNELS = 2

    def __init__(self, file_name):
        rospy.init_node('ros_record')

        self.file_name = file_name

        self.bridge = CvBridge()
        self.latest_depth = None

        self.soundhandle = SoundClient(blocking=False)
        self.audio_player = AudioPlayer(CHIRP_FILE)
        self.audio_recorder = AudioRecorder(channels=self.CHANNELS)

        rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)

        while self.latest_depth is None and not rospy.is_shutdown():
            rospy.loginfo("WAITING FOR CAMERA DATA.")
            rospy.sleep(.1)


        wavs = []
        images = []

        rate = rospy.Rate(self.CHIRP_RATE)
        while not rospy.is_shutdown():
            image = self.bridge.imgmsg_to_cv2(self.latest_depth)
            images.append(image)
            
            self.soundhandle.playWave(CHIRP_FILE)
            #self.audio_player.play() # takes a while to actually play. (about .015 seconds)
            rospy.sleep(.04)
            self.audio_recorder.record(self.RECORD_DURATION)
            while not self.audio_recorder.done_recording():
                rospy.sleep(.005)
            audio = self.audio_recorder.get_data()
            wavs.append(audio[1])
            rate.sleep()


        audio = np.array(wavs)
        depth = np.array(images)

        rospy.loginfo("Saving data to disk...")
        np.savez_compressed(self.file_name, audio=audio, depth=depth)
	#with h5py.File(self.file_name, 'w') as h5:
	#	h5.create_dataset('audio', data=audio)
	#	h5.create_dataset('depth', data=depth)

        #self.audio_player.shutdown()
        self.audio_recorder.shutdown()


    def depth_callback(self, depth_image):
        self.latest_depth = depth_image


if __name__ == "__main__":
    Recorder(sys.argv[1])
  

