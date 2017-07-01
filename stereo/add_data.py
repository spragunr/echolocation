import h5py
import os
import numpy as np
from sys import argv
from stereo_processing import align_audio, downsize

path = '/Volumes/seagate/legit_data/'
current_path = os.getcwd()+'/'
print "opening main file"
with h5py.File(current_path+argv[1], 'r') as main_data:
	main_audio = main_data['audio'].value
	main_depth = main_data['depth'].value
new_audio = [main_audio]
new_depth = [main_depth]
num_new_samples = 0
old_audio_shape = main_audio.shape
old_depth_shape = main_depth.shape
for filename in argv[2:]:
	print "loading %s data" %filename
	with h5py.File(path+filename, 'r') as f:
		print "aligning audio"
		a = f['audio'].value
		aligned = align_audio(5000, a)
		new_audio.append(aligned)
		print "downsizing depth"
		d = f['depth'].value
		downsized = np.empty((aligned.shape[0],12,16))
		counter = 0
		for d_map in d:
			downsized[counter] = downsize(d_map)
			print "done with map", counter
			counter += 1
		new_depth.append(downsized)
		num_new_samples += a.shape[0]
audio_tuple = tuple(new_audio)
depth_tuple = tuple(new_depth)
print "audio concatenation"
all_audio = np.concatenate(audio_tuple)
print "depth concatenation"
all_depth = np.concatenate(depth_tuple)
print "\n\nold audio shape:", old_audio_shape
print "old depth shape:", old_depth_shape
print "total number of new samples added:",num_new_samples 
print "new audio shape:", all_audio.shape
print "new depth shape:", all_depth.shape
print "\n\nsaving new file"
with h5py.File("data_100t.h5", 'w') as d:
	d.create_dataset('audio', data=all_audio)
	d.create_dataset('depth', data=all_depth)

