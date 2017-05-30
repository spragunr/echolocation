"""
stereo version
"""

import numpy as np

from sys import argv

from scipy import signal
import matplotlib.pyplot as plt



def downsize(img):
	# downsize spectrogram
	# 640x480 to 32x24 (downsized by factor of 20)
	# window size: 20x20 

	orig_dims = img.shape
	ds_dims = (orig_dims[0]/20, orig_dims[1]/20)
	downsized_img = np.zeros(ds_dims)
	for i in range(0,orig_dims[0],20):
		for j in range(0,orig_dims[1],20):
			window = img[i:i+1, j:j+1].flatten()
			non_zero = np.delete(window, np.where(window==0))
			if non_zero:
				downsized_img[i,j] = np.mean(non_zero)
			# else, the space remains a zero 
	return downsized_image

def get_data():
:


