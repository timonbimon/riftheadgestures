__author__ = "timon"

import numpy as np

import tensorflow as tf
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn.rnn_cell import BasicLSTMCell, LSTMCell

import read_data

# external flags
flags = tf.app.flags
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('data_dir', '../data', 'Directory to put the data.')
FLAGS = flags.FLAGS

def build_graph():
    # lstm

    # prediction

    # loss

    pass

def train():
    pass

if __name__ == "__main__":
    train()
    print FLAGS.seq_width
