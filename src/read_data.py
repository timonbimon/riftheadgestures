__author__ = "timon"

# inspired by input_data.py from the TensorFlow MNIST data tutorial
# https://www.tensorflow.org/versions/r0.8/tutorials/mnist/download/index.html

import glob
import os
import numpy as np
import tensorflow as tf

"""Read input data from .csv files and create dataset"""

flags = tf.app.flags
# measurements
flags.DEFINE_integer('max_seq_length', 200, 'Maximum length of sequence allwoed')
flags.DEFINE_integer('seq_width', 11, 'Number of features')
# datasets
flags.DEFINE_float('train_split', 0.7, 'Percentag of data used for training')
flags.DEFINE_float('val_split', 0.15, 'Percentag of data used for validation')
flags.DEFINE_float('test_split', 0.15, 'Percentag of data used for testing')
# labels
flags.DEFINE_integer('num_classes', 3, 'Number of different classes')
flags.DEFINE_integer('yes', 0, 'Coded label for YES')
flags.DEFINE_integer('no', 1, 'Coded label for NO')
flags.DEFINE_integer('null', 2, 'Coded label for NULL')
FLAGS = flags.FLAGS

class DataSet(object):
    def __init__(self, measurements, seq_lengths, labels):
        self._measurements = measurements
        self._seq_lengths = seq_lengths
        self._labels = labels
        self._num_meas = measurements.shape[0]
        self._index = 0
        self._epochs = 0

    @property
    def num_meas(self):
        return self._num_meas

    def next_batch(self, batch_size):
        start = self._index
        self._index += batch_size
        if self._index > self._num_meas:
            # Finished epoch
            self._epochs += 1
            # Shuffle the data
            perm = np.arange(self._num_meas)
            np.random.shuffle(perm)
            self._measurements = self._measurements[perm]
            self._seq_lengths = self._seq_lengths[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index = batch_size
            assert batch_size <= self._num_meas
        end = self._index
        return (self._measurements[start:end],
               self._seq_lengths[start:end],
               self._labels[start:end])

def read_data(data_path):
    print "Reading in data..."
    files = glob.glob(os.path.join(data_path, "*.csv"));
    # get num of data points and init numpy arrays
    num_meas = len(files)
    measurements = np.zeros(shape=(num_meas, FLAGS.max_seq_length, FLAGS.seq_width))
    seq_lengths = np.zeros(num_meas)
    labels = np.zeros(num_meas)
    # map labels to numbers
    lab2num = { "YES" : [FLAGS.yes, 0], "NO": [FLAGS.no, 0], "NULL": [FLAGS.null, 0] }
    for i, filename in enumerate(files):
        lab = lab2num[os.path.basename(filename).split('_')[0]]
        labels[i] = lab[0]
        # keep count
        lab[1] += 1

        next_meas = np.genfromtxt(filename, delimiter=',')
        len_seq = next_meas.shape[0]
        if len_seq > FLAGS.max_seq_length:
            print "Sequence is too long: " + str(len_seq)
            len_seq = FLAGS.max_seq_length

        next_meas.resize(FLAGS.max_seq_length, FLAGS.seq_width)
        measurements[i] = next_meas

        seq_lengths[i] = len_seq
    # convert datatype
    measurements.astype(np.float32)
    seq_lengths.astype(np.int32)
    labels.astype(np.int64)
    # permute data
    perm = np.arange(num_meas)
    np.random.shuffle(perm)
    measurements = measurements[perm]
    seq_lengths = seq_lengths[perm]
    labels = labels[perm]
    # output statistic
    print "Read in %d sequences" % num_meas
    print "YES: %d" % lab2num["YES"][1]
    print "NO: %d" % lab2num["NO"][1]
    print "NULL: %d" % lab2num["NULL"][1]
    return (measurements, seq_lengths, labels)

def get_datasets(data_path):
    """Read data and split into train/val/test

    Args:
        data_path: absolute path to data files

    Returns:
        three DataSet objects, namely train, val & test
    """

    measurements, seq_lengths, labels = read_data(data_path)
    num_meas = measurements.shape[0]
    train_split = int(num_meas * FLAGS.train_split)
    val_split = train_split + int(num_meas * FLAGS.val_split)

    train_set = DataSet(measurements[:train_split],
                        seq_lengths[:train_split],
                        labels[:train_split])
    val_set = DataSet(measurements[train_split:val_split],
                        seq_lengths[train_split:val_split],
                        labels[train_split:val_split])
    test_set = DataSet(measurements[val_split:],
                        seq_lengths[val_split:],
                        labels[val_split:])
    return (train_set, val_set, test_set)


if __name__ == "__main__":
    _,_,test_set = get_datasets('../data')
    measurements, seq_length, labs = test_set.next_batch(5)
    print measurements
    print seq_length
    print labs
