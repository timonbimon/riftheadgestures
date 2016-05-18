__author__ = "timon"

# inspired by tensorflow tutorial
# https://github.com/tensorflow/tensorflow/blob/r0.8/tensorflow/examples/tutorials/mnist/fully_connected_feed.py

import tensorflow as tf
import time
import sys

import read_data
import lstm


flags = tf.app.flags
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_string('data_dir', '../data', 'Directory to put the data.')
FLAGS = flags.FLAGS

def fill_feed_dict(data_set, seq_input, early_stop, labels):
    seq_input_feed, early_stop_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)
    feed_dict = {
        seq_input: seq_input_feed,
        early_stop: early_stop_feed,
        labels: labels_feed
    }
    return feed_dict

def do_eval(sess,
            eval_correct,
            seq_input,
            early_stop,
            labels,
            data_set):
  """Runs one evaluation against the full epoch of data."""
  # And run one epoch of eval.
  true_count = 0.0  # Counts the number of correct predictions.
  steps_per_epoch = int(data_set.num_meas / FLAGS.batch_size)
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                                seq_input, early_stop, labels)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = true_count / num_examples
  print('  Num examples: %d  Num correct: %d  Precision: %0.04f' %
        (num_examples, true_count, precision))

def run_training():
    train_set, val_set, test_set = read_data.get_datasets(FLAGS.data_dir)

    # placeholder for one batch of input
    seq_input = tf.placeholder(tf.float32,
                                [FLAGS.batch_size,
                                FLAGS.max_seq_length,
                                FLAGS.seq_width])
    # placeholder for the length of each sequence
    early_stop = tf.placeholder(tf.int32, [FLAGS.batch_size])
    # placeholder for the labels
    labels = tf.placeholder(tf.int64, [FLAGS.batch_size])

    print "Build graph..."
    # Build a Graph that computes predictions from the inference model.
    print "Build inference part..."
    logits = lstm.inference(seq_input, early_stop)
    # Add to the Graph the Ops for loss calculation.
    print "Build loss part..."
    loss = lstm.loss(logits, labels)
    # Add to the Graph the Ops that calculate and apply gradients.
    print "Build optimization part..."
    print "Using learning rate %f" % FLAGS.learning_rate
    train_op = lstm.training(loss, FLAGS.learning_rate)
    # Add the Op to compare the logits to the labels during evaluation.
    print "Build evaluation part..."
    eval_correct = lstm.evaluation(logits, labels)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()
    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    print "Start training..."
    with tf.Session() as sess:
        # Run the Op to initialize the variables.
        init = tf.initialize_all_variables()
        sess.run(init)
        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.train.SummaryWriter(FLAGS.data_dir, sess.graph)

        # And then after everything is built, start the training loop.
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()

            # Fill a feed dictionary with the actual set of measurements,
            # sequence lengths and labels for this particular training step.
            feed_dict = fill_feed_dict(train_set,
                                    seq_input, early_stop, labels)

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.
            _, loss_value = sess.run([train_op, loss],
                                    feed_dict=feed_dict)

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if step % 5 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # Save a checkpoint and evaluate the model
            if (step + 1) == FLAGS.max_steps:
                saver.save(sess, FLAGS.data_dir, global_step=step)
                # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(sess,
                        eval_correct,
                        seq_input,
                        early_stop,
                        labels,
                        train_set)
                # Evaluate against the validation set.
                print('Validation Data Eval:')
                do_eval(sess,
                        eval_correct,
                        seq_input,
                        early_stop,
                        labels,
                        val_set)

def main(argv):
    run_training()

if __name__ == "__main__":
    tf.app.run()
