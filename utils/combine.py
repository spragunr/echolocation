#!/usr/bin/env python  
""" Script for combining multiple .h5 files into one. """

import numpy as np
import h5py
import sys
import os
from collections import defaultdict
import argparse

def total_size(files, path='./'):
    '''
    @PURPOSE: determine the total number of data points in a set of h5 files
    @PARAMS: files - [list of strings] file names
    @RETURN: [int] number of points
    '''
    total = 0
    for name in files:
        with h5py.File(path+name, 'r') as data:
            total += data[data.keys()[0]].shape[0]
    return total

def append_to_set(sets, data, index, total_size, set_name):
    '''
    @PURPOSE: Append preprocessed data to the appropriate set the
              data set will be created if it does not exist.
    @PARAMS: sets - [opened h5 file] data sets
             data - [numpy array] data to append
             index - [int] index to append
             total_size - [int] size of entire data set
             set_name - [string] name of set to append to
    @RETURN: [int] Next index for an append
    '''
    if not set_name in sets:
        set_shape = tuple([total_size] + list(data.shape[1:]))
        sets.create_dataset(set_name, set_shape, dtype=data.dtype, chunks=True)
    sets[set_name][index:index+data.shape[0],...] = data[:]
    return index + data.shape[0]

def combine_all(in_files, out_name):
 
    data_length = total_size(in_files, "")
    index_dict = defaultdict(lambda: 0)
    out_h5 = h5py.File(out_name, 'w')
    out_h5.create_dataset('file_names', data=in_files, dtype="S128")
    file_starts = np.zeros(len(in_files), dtype=np.int64)
    for i, file_name in enumerate(in_files):
        print file_name
        in_h5 = h5py.File(file_name, 'r')
        for key in in_h5:
            print key
            file_starts[i] = index_dict[key]
            index_dict[key] = append_to_set(out_h5, in_h5[key],
                                            index_dict[key], data_length, key)

    out_h5.create_dataset('file_starts', data=file_starts, dtype=np.int64)
            
    out_h5.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('h5_files', nargs='*')
    parser.add_argument('output_file')
    args = parser.parse_args()
    combine_all(args.h5_files, args.output_file)
    
    
if __name__ == "__main__":
    main()
    # files = ['downHall_train1.h5', 'downHall_train2.h5',
    #          'forensics_train1.h5', 'forensics_train2.h5',
    #          'isat250_train1.h5', 'isat250_train2.h5',
    #          'isat267_train1.h5', 'isat267_train2.h5',
    #          'mainHall_train1.h5', 'mainHall_train2.h5',
    #          'officeHall_train1.h5', 'officeHall_train2.h5',
    #          'office_train1.h5', 'office_train2.h5',
    #          'robot_train1.h5', 'robot_train2.h5',
    #          'stair_train1.h5', 'stair_train2.h5',
    #          'upHall_train1.h5', 'upHall_train2.h5']

    # files = ['downHall_test.h5', 'forensics_test.h5',
    #          'isat250_test.h5', 'isat267_test.h5', 'mainHall_test.h5',
    #          'officeHall_test.h5', 'office_test.h5', 'robot_test.h5',
    #          'stair_test.h5', 'upHall_test.h5']
    
    # combine_all(files, "test100k.h5")
    
