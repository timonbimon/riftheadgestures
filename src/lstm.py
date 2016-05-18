__author__ = "timon"

# inspired by tensorflows tutorial
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py

import numpy as np
import math

import tensorflow as tf
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn.rnn_cell import LSTMCell

import read_data

# external flags
flags = tf.app.flags
flags.DEFINE_integer('batch_size', 10, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('num_units', 1, 'Number of units in one LSTM cell')
FLAGS = flags.FLAGS

def inference(seq_input, early_stop):
    lstm_cell = tf.nn.rnn_cell.LSTMCell(FLAGS.num_units, FLAGS.seq_width,
                                        initializer=tf.random_uniform_initializer(-1,1))
    initial_state = lstm_cell.zero_state(FLAGS.batch_size, tf.float32)

    # cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)
    cell = lstm_cell

    # inputs for rnn needs to be a list, each item being a timestep.
    # we need to split our input into each timestep, and reshape it because split keeps dims by default
    inputs = [tf.reshape(i, (FLAGS.batch_size, FLAGS.seq_width)) for i in tf.split(1, FLAGS.max_seq_length, seq_input)]
    outputs, state = tf.nn.rnn(cell, inputs, initial_state=initial_state, sequence_length=early_stop)    # prediction

    # softmax layer
    output = tf.reshape(tf.concat(1, outputs), [FLAGS.batch_size, FLAGS.num_units * FLAGS.max_seq_length])
    softmax_w = tf.get_variable("softmax_w", [FLAGS.num_units * FLAGS.max_seq_length, FLAGS.num_classes],
                                initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(FLAGS.num_units))))
    softmax_b = tf.get_variable("softmax_b", [FLAGS.num_classes],
                                initializer=tf.constant_initializer(0.0))
    logits = tf.matmul(output, softmax_w) + softmax_b
    return logits


def training(loss, learning_rate):
    # Add a scalar summary for the snapshot loss.
    tf.scalar_summary(loss.op.name, loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits, labels, name='cross_entropy')
    loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
    return loss


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))

if __name__ == "__main__":
    pass
