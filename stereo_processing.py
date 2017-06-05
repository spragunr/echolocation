import h5py
import numpy as np
import os 
import os.path

from scipy import signal

######################################################

def downsize(img):
	# downsize depth map 
	# 480x640 to 12x16 (downsized by factor of 40)
	# window size: 40x40 

	orig_dims = img.shape
	ds_dims = (orig_dims[0]/40, orig_dims[1]/40)
	downsized_img = np.zeros(ds_dims)
	for i in range(0,orig_dims[0],40):
		for j in range(0,orig_dims[1],40):
			window = img[i:i+40, j:j+40].flatten()
			non_zero = np.delete(window, np.where(window==0))
			if non_zero.size != 0:
				downsized_img[i/40,j/40] = np.mean(non_zero)
#			# ask sprague about this
#			else: 
#				downsized_img[i/40,j/40] = 0.00000000001 

	return downsized_img

######################################################

def get_data():
	files = ['forensics', 'isat243', 'isat246', 'isat248', 'office', 'stairwell', 'spine'] 
	audio_list = []
	depth_list = []
	path = os.getcwd() 
	print "\n"
	for i in range(len(files)):
		print "loading '%s' data..." %files[i]
		with np.load(path+'/data_stereo/'+files[i]+'.npz') as d:
			audio_list.append(d['audio'])
			depth_list.append(d['depth'])
	print "---------------------------------"
	print "data loading complete\n"

	print "concatenating audio data..."
	audio_tuple = tuple(audio_list)
	audio = np.concatenate(audio_tuple)

	print "downsizing depth data..."
	new_depth = np.empty((audio.shape[0],12,16))
	counter = 0
	for d_file in depth_list:
		for d_map in d_file:
			new_depth[counter] = downsize(d_map)
			counter += 1

	print "saving concatenated audio and downsized depth data saved as 'all.h5'...\n"
	with h5py.File('all.h5', 'w') as hf:
		hf.create_dataset('audio', data=audio)
		hf.create_dataset('depth', data=new_depth)

######################################################

def preprocess_data():
	print "preprocessing data..."
	path = os.getcwd()
	with h5py.File(path+'/all.h5', 'r') as data:
		audio = data['audio'][:]	
		depth = data['depth'][:] # shape: 13274, 12, 16

	depth_reshaped = np.reshape(depth, (depth.shape[0],-1)) # shape: 13274, 192

	if os.path.isfile('input_spectrograms.h5'):
		print "fetching spectrogram array from 'input_spectrograms.h5'..."
		with h5py.File(path+'/input_spectrograms.h5', 'r') as sgrams:
			input_set = sgrams['spectrograms'][:]
	else: 
		print "creating spectrogram set 0" 
		freq1, time1, spectro1 = signal.spectrogram(audio[0,:,0], noverlap=250)
		freq2, time2, spectro2 = signal.spectrogram(audio[0,:,1], noverlap=250)
		dims = spectro1.shape
		input_set = np.empty((audio.shape[0], dims[0], dims[1], 2))
		input_set[0,:,:,0] = spectro1
		input_set[0,:,:,1] = spectro2
	
		for i in range(1,audio.shape[0]):
			print "creating spectrogram set", i
			freq1, time1, spectro1 = signal.spectrogram(audio[i,:,0], noverlap=250)
			freq2, time2, spectro2 = signal.spectrogram(audio[i,:,1], noverlap=250)
			input_set[i,:,:,0] = spectro1
			input_set[i,:,:,1] = spectro2
		print "saving array of spectrograms as 'input_spectrograms.h5'..."
		with h5py.File('input_spectrograms.h5', 'w') as sgrams:
			sgrams.create_dataset('spectrograms', data=input_set)
	
	return input_set, depth_reshaped

######################################################

def split_data(x, y):
	if not os.path.isfile('model_sets.h5'):
		print "splitting dataset into training and test sets..."
		xshape = x.shape[0]
		xbool = np.zeros_like(x, dtype=bool)
		ybool = np.zeros_like(y, dtype=bool)
		subset_size = int(xshape/7)
		half_test_size = int((xshape*0.2)/14)
	
		for i in range(7):
			print "creating bool array for subset",i
			start_index = subset_size*i
			end_index = start_index + subset_size
			mid_index = int((0.5*(end_index-start_index)) + start_index) 
			test_start = mid_index-half_test_size
			test_end = mid_index+half_test_size 
			xbool[test_start:test_end] = True
			ybool[test_start:test_end] = True 

		print "reshaping test vectors..."
		xtest = x[xbool].reshape((-1,129,385,2))
		ytest = y[ybool].reshape((-1,192))
		print "reshaping training vectors..."
		xtrain = x[np.logical_not(xbool)].reshape((-1,129,385,2))
		ytrain = y[np.logical_not(ybool)].reshape((-1,192))
		print "saving data sets..."
		with h5py.File('model_sets.h5', 'w') as sets:
			sets.create_dataset('xtrain', data=xtrain)
			sets.create_dataset('ytrain', data=ytrain)
			sets.create_dataset('xtest', data=xtest)
			sets.create_dataset('ytest', data=ytest)
		print "training and test sets saved as 'model_sets.h5'"
															
#####################################################

def main():
	if not os.path.isfile('all.h5'):
		get_data()
	input_set, target = preprocess_data()
	split_data(input_set, target)

main()

