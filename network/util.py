import numpy as np
import tensorflow as tf

import keras.backend as K
import keras
import h5py

import cross


# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
class DataGenerator(keras.utils.Sequence):
    def __init__(self, file_name, h5_audio, h5_depth, batch_size,
                 sound_time=.06, shift=.02, no_shift=False, noise=.0,
                 shuffle=False, tone_noise=.05, flip=False):
        self.file_name = file_name
        self.batch_size = batch_size
        self.audio_length = int(44100 * sound_time)
        self.shift = shift
        self.no_shift = no_shift
        self.noise = noise
        self.shuffle = shuffle
        self.tone_noise = tone_noise
        self.flip = flip
        self.h5_audio = h5_audio
        self.h5_depth = h5_depth
        sets = h5py.File(self.file_name, 'r')
        h5_audio = sets[self.h5_audio]
        self.num_samples = h5_audio.shape[0]
        sets.close()
        self.on_epoch_end()

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
       
        self.indexes = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indexes)
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(self.num_samples // self.batch_size)

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch

        # h5py is not multiprocess safe, so this needs to be opened in
        # every call to getitem. 
        sets = h5py.File(self.file_name, 'r')
        h5_audio = sets[self.h5_audio]
        h5_depth = sets[self.h5_depth]
        
        frm = index * self.batch_size
        to = (index + 1) * self.batch_size
        indices = sorted(self.indexes[frm:to]) # h5py requires ordered access

        sample_length = self.audio_length
        result_length = int(sample_length * (1 - self.shift))

        if self.no_shift:
            start_ind = np.zeros(self.batch_size, dtype='int32')
        else:
            start_ind = np.random.randint(sample_length - result_length + 1,
                                          size=self.batch_size)
            
        x_data_left = np.empty((self.batch_size, result_length, 1), dtype='float32')
        x_data_right = np.empty((self.batch_size, result_length, 1), dtype='float32')
        y_data = h5_depth[indices, ...] / 1000.

        left_channel = 0
        right_channel = 1

        if self.flip and np.random.random() < .5:
            left_channel = 1
            right_channel = 0
            if len(y_data.shape) > 2: # depth map
                y_data = y_data[:, : , ::-1, ...]
            else: # 3d points
                y_data[:, 0] *= -1.0

        for i, index in enumerate(indices):
            frm = start_ind[i]
            to = start_ind[i] + result_length
            x_data_left[i, :, 0] = h5_audio[index, frm:to, left_channel]
            x_data_right[i, :, 0] = h5_audio[index, frm:to, right_channel]
        x_data_left /= 32000.
        x_data_right /= 32000.

                
         # Add a sin wave with random freqency and phase...
        if self.tone_noise > 0:
            t = np.linspace(0, result_length /44100., result_length)
            for i in range(self.batch_size):
                #pitch = np.random.random() * 115.0 + 20 # midi pitch
                pitch = np.random.random() * 88 + 20 # midi pitch max (c8)
                amp = np.random.random() * self.tone_noise
                #people.sju.edu/~rhall/SoundingNumber/pitch_and_frequency.pdf 
                freq = 440 * 2**((pitch - 69)/12.)
                phase = np.pi * 2 * np.random.random()
                tone = np.sin(2 * np.pi * freq * t + phase) * amp
                x_data_left[i, :, 0] += tone
                x_data_right[i, :, 0] += tone
                
        # Random multiplier for all samples...
        x_data_left *= (1. + np.random.randn(*x_data_left.shape) * self.noise)
        x_data_right *= (1. + np.random.randn(*x_data_right.shape) * self.noise)

        # Calculate cross-correlations
        num_crosses = 192
        length_crosses = 32
        span = 21
        num_channels = 10
        xcr_data = np.empty((self.batch_size, num_crosses, span, num_channels), dtype='float32')
        # for i in range(self.batch_size):
        #     xcr_data[i, ...] = cross.xcr_chirp(x_data_left[i, :, 0],
        #                                        x_data_right[i, :, 0],
        #                                        num_crosses,
        #                                        length_crosses, span,
        #                                        num_channels)
        
        # xcr_data = np.log(xcr_data + 1.0)
        #xcr_data = np.log(xcr_data + 1.0) / 8.0
        
        #cross.show_xcrs(xcr_data)
            
        sets.close()
        return [x_data_left, x_data_right, xcr_data], y_data
        #return xcr_data, y_data
        
def raw_generator(x_train, xcr_train, y_train, batch_size=64,
                  shift=.02, no_shift=False, noise=.00,
                  shuffle=True, tone_noise=.05, flip=False):
    num_samples = x_train.shape[0]
    sample_length = x_train.shape[1]
    result_length = int(sample_length * (1 - shift))
    batch_index = 0
        
    while True:
        # Shuffle before each epoch
        if batch_index == 0 and shuffle:
            indices = np.random.permutation(x_train.shape[0])
            np.take(x_train,indices,axis=0,out=x_train)
            np.take(xcr_train,indices,axis=0,out=xcr_train)
            np.take(y_train,indices,axis=0,out=y_train)

        # Randomly crop the audio data...
        if no_shift:
            start_ind = np.zeros(batch_size, dtype='int32')
        else:
            start_ind = np.random.randint(sample_length - result_length + 1,
                                          size=batch_size)
            
        x_data_left = np.empty((batch_size, result_length, 1))
        x_data_right = np.empty((batch_size, result_length, 1))

        xcr_data = xcr_train[batch_index:batch_index + batch_size, ...]

        y_data = y_train[batch_index:batch_index + batch_size, ...]

        left_channel = 0
        right_channel = 1
        if flip and np.random.random() < .5:
            left_channel = 1
            right_channel = 0
            if len(y_data.shape) > 2: # depth map
                y_data = y_data[:, : , ::-1, ...]
            else: # 3d points
                y_data[:, 0] *= -1.0

            # import matplotlib.pyplot as plt
            # plt.imshow(xcr_data[0,:,:,0])
            # plt.figure()
                       
            xcr_data = xcr_data[:,:,::-1,:]
            # plt.imshow(xcr_data[0,:,:,0])
            # plt.show()
                
        
        for i in range(batch_size):
            x_data_left[i, :, 0] = x_train[batch_index + i,
                                           start_ind[i]:start_ind[i] + result_length,
                                           left_channel]
            x_data_right[i, :, 0] = x_train[batch_index + i,
                                            start_ind[i]:start_ind[i] + result_length,
                                            right_channel]

        # Add a sin wave with random freqency and phase...
        if tone_noise > 0:
            t = np.linspace(0, result_length /44100., result_length)
            for i in range(batch_size):
                #pitch = np.random.random() * 115.0 + 20 # midi pitch
                pitch = np.random.random() * 88 + 20 # midi pitch max (c8)
                amp = np.random.random() * tone_noise
                #people.sju.edu/~rhall/SoundingNumber/pitch_and_frequency.pdf 
                freq = 440 * 2**((pitch - 69)/12.)
                phase = np.pi * 2 * np.random.random()
                tone = np.sin(2 * np.pi * freq * t + phase) * amp
                x_data_left[i, :, 0] += tone
                x_data_right[i, :, 0] += tone


        
            
        # Random multiplier for all samples...
        x_data_left *= (1. + np.random.randn(*x_data_left.shape) * noise)
        x_data_right *= (1. + np.random.randn(*x_data_right.shape) * noise)
        #x_data_left += np.random.randn(*x_data_left.shape) * noise
        #x_data_right += np.random.randn(*x_data_right.shape) * noise

        batch_index += batch_size

        if batch_index > (num_samples - batch_size):
            batch_index = 0

        yield [x_data_left, x_data_right, xcr_data], y_data


def validation_split(data, split=.1, chunk_size=200):
    """Subsequent data points are very highly correlated.  This means that
    pulling out a validation set randomly from the training data will
    not give useful validation: it is likely that there will be a data
    point in the training set that is almost identical to the
    validation data.  This method splits out validation data in chunks
    to alleviate this issue.

    """
    val_size = int(split * data.shape[0])
    chunks = val_size // chunk_size
    val_size = chunk_size * chunks
    train_size = data.shape[0] - val_size
    block_size = data.shape[0] // chunks
    train_chunk_size = block_size - chunk_size

    val = np.empty([val_size] + list(data.shape[1::]), dtype=data.dtype)
    train = np.empty([train_size] + list(data.shape[1::]), dtype=data.dtype)

    for i in range(chunks):
        # indices in the original data:
        block_start = i * block_size
        chunk_end = block_start + chunk_size

        # indices in the validation set:
        start = i * chunk_size
        end = start + chunk_size

        val[start:end, ...] = data[block_start:chunk_end, ...]

        # indices in the original data:
        train_start = chunk_end
        train_end = block_start + block_size

        # indices in the train set:
        start = i * train_chunk_size
        end = start + train_chunk_size

        train[start:end, ...] = data[train_start:train_end, ...]

    # grab any partial final training data
    leftover =  data.shape[0] % chunks
    if leftover > 0:
        train[-leftover:, ...] = data[-leftover:, ...]

    return val, train

#####################################################
# LOSS FUNCTIONS
#####################################################

def safe_mse(y_true, y_pred):
    # reshape so that targets can be any shape
    batch_size = tf.shape(y_true)[0]
    y_true = tf.reshape(y_true, (batch_size, -1))
    y_pred = tf.reshape(y_pred, (batch_size, -1))

    zero = tf.constant(0, dtype=K.floatx())
    ok_entries = tf.not_equal(y_true, zero)
    safe_targets = tf.where(ok_entries, y_true, y_pred)
    sqr = tf.square(y_pred - safe_targets)
    
    valid = tf.cast(ok_entries, K.floatx())
    num_ok = tf.reduce_sum(valid, axis=-1) # count OK entries
    num_ok = tf.maximum(num_ok, tf.ones_like(num_ok)) # avoid divide by zero
    
    return tf.reduce_sum(sqr, axis=-1) / num_ok

def safe_berhu(y_true, y_pred):
    # reshape so that targets can be any shape
    batch_size = tf.shape(y_true)[0]
    y_true = tf.reshape(y_true, (batch_size, -1))
    y_pred = tf.reshape(y_pred, (batch_size, -1))

    zero = tf.constant(0, dtype=K.floatx())
    ok_entries = tf.not_equal(y_true, zero)
    safe_targets = tf.where(ok_entries, y_true, y_pred)
    
    diffs = y_pred - safe_targets
    abs_diffs = tf.abs(diffs)

    fifth = tf.constant(1./5., dtype=K.floatx())
    c = fifth * tf.reduce_max(abs_diffs)
    
    l2_diffs = (tf.square(diffs) + c**2) / (2 * c)

    combined = tf.where(abs_diffs < c, abs_diffs, l2_diffs)

    valid = tf.cast(ok_entries, K.floatx())
    num_ok = tf.reduce_sum(valid, axis=-1) # count OK entries
    num_ok = tf.maximum(num_ok, tf.ones_like(num_ok)) # avoid divide by zero
    return tf.reduce_sum(combined, axis=-1) / num_ok


def safe_l1(y_true, y_pred):
    # reshape so that targets can be any shape
    batch_size = tf.shape(y_true)[0]
    y_true = tf.reshape(y_true, (batch_size, -1))
    y_pred = tf.reshape(y_pred, (batch_size, -1))

    zero = tf.constant(0, dtype=K.floatx())
    ok_entries = tf.not_equal(y_true, zero)
    safe_targets = tf.where(ok_entries, y_true, y_pred)
    
    diffs = y_pred - safe_targets
    abs_diffs = tf.abs(diffs)

    valid = tf.cast(ok_entries, K.floatx())
    num_ok = tf.reduce_sum(valid, axis=-1) # count OK entries
    num_ok = tf.maximum(num_ok, tf.ones_like(num_ok)) # avoid divide by zero
    return tf.reduce_sum(abs_diffs, axis=-1) / num_ok


# SSIM implementation based on:

""" https://stackoverflow.com/questions/39051451/ssim-ms-ssim-for-tensorflow """
""" https://github.com/mrharicot/monodepth/blob/228c98dc3ed075f8e8325aa993ed36dcf8a356e8/monodepth_model.py """

def l1_ssim(img1, img2, alpha=.9):
    ss = ssim_multi(img1, img2)
    l = K.mean(safe_l1(img1, img2))
    return alpha * ss + (1.0 - alpha) * l
    #return alpha * ssim_multi(img1, img2) + (1.0 - alpha) * safe_l1(img1, img2)

def berhu_ssim(img1, img2, alpha=.5):
    ss = ssim_multi(img1, img2)
    l = K.mean(safe_berhu(img1, img2))
    return alpha * ss + (1.0 - alpha) * l

def ssim_multi(img1, img2, sizes=(3,5,7,11), weights=(.25,.25,.25,.25)):
    total = []
    for size,weight in zip(sizes, weights):
        total += [ssim(img1, img2, size) * weight]
    total = K.stack(total)
    #mean = tf.reduce_sum(tf.boolean_mask(total, tf.logical_not(tf.is_nan(total))))
    return K.sum(total)


def ssim(img1, img2, size=11):
    img1 = tf.expand_dims(img1, -1)
    img2 = tf.expand_dims(img2, -1)
    zero_locs = K.pool2d(-img1, (size,size), strides=(1,1), padding="valid", pool_mode="max")
    K1 = 0.01
    K2 = 0.03
    L = 10.0 # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = K.pool2d(img1, (size,size), strides=(1,1), padding="valid", pool_mode="avg")
    mu2 = K.pool2d(img2, (size,size), strides=(1,1), padding="valid", pool_mode="avg")

    sigma_1 = K.pool2d(img1**2, (size,size), strides=(1,1), padding="valid", pool_mode="avg") - mu1**2
    sigma_2 = K.pool2d(img2**2, (size,size), strides=(1,1), padding="valid", pool_mode="avg") - mu2**2
    sigma_12 = K.pool2d(img1 * img2, (size,size), strides=(1,1), padding="valid", pool_mode="avg") - mu1 * mu2

    SSIM_n = (2 * mu1 * mu2 + C1) * (2* sigma_12 + C2)
    SSIM_d = (mu1**2 + mu2**2 + C1) * (sigma_1 + sigma_2 + C2)

    
    SSIM = SSIM_n / SSIM_d
    SSIM_loss = (1-SSIM) / 2.0
    SSIM_loss = tf.where(tf.equal(zero_locs,tf.zeros_like(SSIM_loss)),
                         tf.zeros_like(SSIM_loss), SSIM_loss)

    return K.sum(SSIM_loss)/ tf.cast(tf.count_nonzero(SSIM_loss), dtype=tf.float32)

def berhu_test():
    tf.enable_eager_execution()
    x = np.array(np.random.random((3,4,5)), dtype='float32')
    y = np.array(np.random.random((3,4,5)), dtype='float32')
    print safe_berhu(x,y)
    print safe_mse(x,y)
    print safe_l1(x,y)
    x = x.reshape(-1, 20)
    y = y.reshape(-1, 20)
    print safe_berhu(x,y)
    print safe_mse(x,y)
    print safe_l1(x,y)

def ssim_test():
    import numpy as np
    import tensorflow as tf
    from skimage import data, img_as_float
    import matplotlib.pyplot as plt
    import h5py

    tf.enable_eager_execution()

    f = h5py.File('/home/spragunr/prepped100k_w_xcr.h5','r')

    true = f['test_depths'][100,...]/1000.
    true = np.expand_dims(true,0)
    pred = f['test_depths'][102,...]/1000.
    pred = np.expand_dims(pred,0)


    plt.imshow(true[0,:,:], interpolation='none')
    plt.show()
    print ssim(true, pred, size=11)
    
if __name__ == "__main__":
    berhu_test()
