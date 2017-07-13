
import numpy as np
import h5py
import sys
from scipy import signal
import os

SAMPLE_RATE = 44100

def align_clip(clip, trigger_freq=16000, threshold=1250,
               box_width=50, percent_required=.5, offset=200):
    w = 5.0
    s = 1.0
    M = np.floor(2. * w * s * SAMPLE_RATE / trigger_freq)
    wavelet = signal.morlet(M, w=5.0, s=1.0, complete=True)
    resp = np.abs(signal.convolve(clip[:,0], wavelet, mode='same'))
    box = np.array([1.0 for _ in range(box_width)])
    above = resp > threshold
    counts = signal.convolve(above, box,mode='valid')
    counts = np.append(counts, np.zeros(box.shape[0] - 1))
    candidates = np.logical_and(above, counts > (percent_required * box_width))
    if np.where(candidates)[0].size > 0:
        start_index = max(0, np.where(candidates)[0][0] - offset)
    else:
        start_index = 0
        print "BAD SAMPLE?"
    result = np.zeros(clip.shape, dtype=clip.dtype)
    result[0:clip.shape[0] - start_index, :] = clip[start_index::, :]
    return result


def compress_h5(file_name):
    data = h5py.File(file_name, 'r')
    compressed = h5py.File(file_name+".h5", 'w')
    for item in data.items():
        dset = compressed.create_dataset(item[0], data[item[0]].shape,
                                         dtype=data[item[0]].dtype,
                                         compression="lzf")
        dset[...] = data[item[0]][...]
    compressed.close()
        

def align_h5(file_name):
    """ Add alligned audio data to an existing h5 file. """
    data = h5py.File(file_name, 'r+')
    if 'audio_aligned' in data:
        del data['audio_aligned']
    dset = data.create_dataset("audio_aligned", data['audio'].shape,
                               dtype=data['audio'].dtype)
    for i in range(data['audio'].shape[0]):
        dset[i, ...] = align_clip(data['audio'][i,...])
    data.close()

def demo():
    """ show example of an alignment """
    import matplotlib.pyplot as plt

    data = h5py.File(sys.argv[1], 'r')

    # Sample rate and desired cutoff frequencies (in Hz).

    clip = data['audio'][100, ...]
    plt.subplot(4,1,1)
    f, t, Sxx = signal.spectrogram(clip[:,0], 44100,
                                   nperseg=256,
                                   noverlap =255)
    plt.pcolormesh(t, f, np.log(1 + Sxx))
    plt.axis('tight')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')


    plt.subplot(4,1,2)

    plt.plot(clip[:,0])
    plt.axis('tight')
    

    plt.subplot(4,1,3)

    aligned = align_clip(clip)
    
    f, t, Sxx = signal.spectrogram(aligned[:,0], 44100,
                                   nperseg=256,
                                   noverlap=255)
    plt.pcolormesh(t, f, np.log(1 + Sxx))
    plt.axis('tight')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')


    
    plt.subplot(4,1,4)
    plt.plot(aligned[:,0])
    plt.axis('tight')
    
    plt.show()


def compress_all():
    files = [#'isat143a', 'isat143b',
             'isat231a', 'isat231b',
             'isat243a', 'isat243b', 'isat246a', 'isat246b',
             'isat246c', 'isat248a', 'isat248b', 'isat248c',
             'isat250a', 'isat250b', 'roboA', 'roboB', 'roboC',
             'roboD', 'roboE']

    for f in files:
        print f
        compress_h5(f)
        os.remove(f)
        

    
    
if __name__ == "__main__":
    #demo()
    #compress_all()
    #compress_h5(sys.argv[1])
    align_h5(sys.argv[1])
