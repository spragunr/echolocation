import numpy as np
import tensorflow as tf

import keras.backend



def raw_generator(x_train, y_train, batch_size=64,
                  shift=.02,no_shift=False, noise=.00,
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
            np.take(y_train,indices,axis=0,out=y_train)

        # Randomly crop the audio data...
        if no_shift:
            start_ind = np.zeros(batch_size, dtype='int32')
        else:
            start_ind = np.random.randint(sample_length - result_length + 1,
                                          size=batch_size)
            
        x_data_left = np.empty((batch_size, result_length, 1))
        x_data_right = np.empty((batch_size, result_length, 1))

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

        yield [x_data_left, x_data_right], y_data



def validation_split_by_chunks(x_train, y_train, split=.1, chunk_size=200):
    """Subsequent data points are very highly correlated.  This means that
    pulling out a validation set randomly from the training data will
    not give useful validation: it is likely that there will be a data
    point in the training set that is almost identical to the
    validation data.  This method splits out validation data in chunks
    to alleviate this issue.

    """
    val_size = int(split * x_train.shape[0])
    chunks = val_size // chunk_size
    val_size = chunk_size * chunks
    train_size = x_train.shape[0] - val_size
    block_size = x_train.shape[0] // chunks
    train_chunk_size = block_size - chunk_size

    x_val = np.empty([val_size] + list(x_train.shape[1::]))
    y_val = np.empty([val_size] + list(y_train.shape[1::]))

    x_train_out = np.empty([train_size] + list(x_train.shape[1::]))
    y_train_out = np.empty([train_size] + list(y_train.shape[1::]))

    for i in range(chunks):
        # indices in the original data:
        block_start = i * block_size
        chunk_end = block_start + chunk_size

        # indices in the validation set:
        start = i * chunk_size
        end = start + chunk_size

        x_val[start:end, ...] = x_train[block_start:chunk_end, ...]
        y_val[start:end, ...] = y_train[block_start:chunk_end, ...]

        # indices in the original data:
        train_start = chunk_end
        train_end = block_start + block_size

        # indices in the train set:
        start = i * train_chunk_size
        end = start + train_chunk_size

        x_train_out[start:end, ...] = x_train[train_start:train_end, ...]
        y_train_out[start:end, ...] = y_train[train_start:train_end, ...]

    # grab any partial final training data
    leftover =  x_train.shape[0] % chunks
    if leftover > 0:
        x_train_out[-leftover:, ...] = x_train[-leftover:, ...]
        y_train_out[-leftover:, ...] = y_train[-leftover:, ...]

    return x_val, y_val, x_train_out, y_train_out


#####################################################

def safe_mse(y_true, y_pred):
    # reshape so that targets can be any shape
    batch_size = tf.shape(y_true)[0]
    y_true = tf.reshape(y_true, (batch_size, -1))
    y_pred = tf.reshape(y_pred, (batch_size, -1))

    zero = tf.constant(0, dtype=keras.backend.floatx())
    ok_entries = tf.not_equal(y_true, zero)
    safe_targets = tf.where(ok_entries, y_true, y_pred)
    sqr = tf.square(y_pred - safe_targets)
    
    valid = tf.cast(ok_entries, keras.backend.floatx())
    num_ok = tf.reduce_sum(valid, axis=-1) # count OK entries
    num_ok = tf.maximum(num_ok, tf.ones_like(num_ok)) # avoid divide by zero
    
    return tf.reduce_sum(sqr, axis=-1) / num_ok

def safe_berhu(y_true, y_pred):
    # reshape so that targets can be any shape
    batch_size = tf.shape(y_true)[0]
    y_true = tf.reshape(y_true, (batch_size, -1))
    y_pred = tf.reshape(y_pred, (batch_size, -1))

    zero = tf.constant(0, dtype=keras.backend.floatx())
    ok_entries = tf.not_equal(y_true, zero)
    safe_targets = tf.where(ok_entries, y_true, y_pred)
    
    diffs = y_pred - safe_targets
    abs_diffs = tf.abs(diffs)

    fifth = tf.constant(1./5., dtype=keras.backend.floatx())
    c = fifth * tf.reduce_max(abs_diffs)
    
    l2_diffs = (tf.square(diffs) + c**2) / (2 * c)

    combined = tf.where(abs_diffs < c, abs_diffs, l2_diffs)

    valid = tf.cast(ok_entries, keras.backend.floatx())
    num_ok = tf.reduce_sum(valid, axis=-1) # count OK entries
    num_ok = tf.maximum(num_ok, tf.ones_like(num_ok)) # avoid divide by zero
    return tf.reduce_sum(combined, axis=-1) / num_ok


def safe_l1(y_true, y_pred):
    # reshape so that targets can be any shape
    batch_size = tf.shape(y_true)[0]
    y_true = tf.reshape(y_true, (batch_size, -1))
    y_pred = tf.reshape(y_pred, (batch_size, -1))

    zero = tf.constant(0, dtype=keras.backend.floatx())
    ok_entries = tf.not_equal(y_true, zero)
    safe_targets = tf.where(ok_entries, y_true, y_pred)
    
    diffs = y_pred - safe_targets
    abs_diffs = tf.abs(diffs)

    valid = tf.cast(ok_entries, keras.backend.floatx())
    num_ok = tf.reduce_sum(valid, axis=-1) # count OK entries
    num_ok = tf.maximum(num_ok, tf.ones_like(num_ok)) # avoid divide by zero
    return tf.reduce_sum(abs_diffs, axis=-1) / num_ok

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
    
if __name__ == "__main__":
    berhu_test()
