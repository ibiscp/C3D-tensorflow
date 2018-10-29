# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
import os
import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.training import checkpoint_utils

import c3d_model
import numpy as np
import cv2
import generate_tfrecords
import sys
from tqdm import tqdm
import argparse

# Basic model parameters
gpu_num = 1
MOVING_AVERAGE_DECAY = 0.9999
model_save_dir = './models'

def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.

    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.

    Args:
      batch_size: The batch size will be baked into both placeholders.

    Returns:
      images_placeholder: Images placeholder.
      labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                           c3d_model.NUM_FRAMES_PER_CLIP,
                                                           c3d_model.CROP_SIZE,
                                                           c3d_model.CROP_SIZE,
                                                           c3d_model.CHANNELS))
    labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
    return images_placeholder, labels_placeholder

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def tower_loss(name_scope, logit, labels):
    cross_entropy_mean = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logit)
    )
    tf.summary.scalar(
        name_scope + '_cross_entropy',
        cross_entropy_mean
    )
    weight_decay_loss = tf.get_collection('weightdecay_losses')
    tf.summary.scalar(name_scope + '_weight_decay_loss', tf.reduce_mean(weight_decay_loss) )

    # Calculate the total loss for the current tower.
    total_loss = cross_entropy_mean + weight_decay_loss
    tf.summary.scalar(name_scope + '_total_loss', tf.reduce_mean(total_loss) )
    return total_loss

def tower_acc(logit, labels):
    correct_pred = tf.equal(tf.argmax(logit, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _variable_with_weight_decay(name, shape, wd):
    var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
    if wd is not None:
        weight_decay = tf.nn.l2_loss(var)*wd
        tf.add_to_collection('weightdecay_losses', weight_decay)
    return var

def run_training(epochs, batch_size, evaluate_every, use_pretrained_model):
    # Get the sets of images and labels for training, validation, and
    # Tell TensorFlow that the model will be built into the default Graph.

    # Create model directory
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    model_filename = "model/sports1m_finetuning_ucf101.model"
    # model_filename = tf.train.latest_checkpoint("models/")

    with tf.Graph().as_default():
        global_step = tf.get_variable(
            'global_step',
            [],
            initializer=tf.constant_initializer(0),
            trainable=False
        )
        images_placeholder, labels_placeholder = placeholder_inputs(
            batch_size * gpu_num
        )
        tower_grads1 = []
        tower_grads2 = []
        logits = []
        opt_stable = tf.train.AdamOptimizer(1e-4)
        opt_finetuning = tf.train.AdamOptimizer(1e-3)
        with tf.variable_scope('var_name') as var_scope:
            weights = {
                'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.0005),
                'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.0005),
                'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.0005),
                'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.0005),
                'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.0005),
                'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.0005),
                'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.0005),
                'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.0005),
                'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.0005),
                'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.0005),
                'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.0005)
            }
            biases = {
                'bc1': _variable_with_weight_decay('bc1', [64], 0.000),
                'bc2': _variable_with_weight_decay('bc2', [128], 0.000),
                'bc3a': _variable_with_weight_decay('bc3a', [256], 0.000),
                'bc3b': _variable_with_weight_decay('bc3b', [256], 0.000),
                'bc4a': _variable_with_weight_decay('bc4a', [512], 0.000),
                'bc4b': _variable_with_weight_decay('bc4b', [512], 0.000),
                'bc5a': _variable_with_weight_decay('bc5a', [512], 0.000),
                'bc5b': _variable_with_weight_decay('bc5b', [512], 0.000),
                'bd1': _variable_with_weight_decay('bd1', [4096], 0.000),
                'bd2': _variable_with_weight_decay('bd2', [4096], 0.000),
                'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.000),
            }

        for gpu_index in range(0, gpu_num):
            with tf.device('/gpu:%d' % gpu_index):

                varlist2 = [ weights['out'],biases['out'] ]
                varlist1 = list((set(weights.values()) | set(biases.values())) - set(varlist2))
                logit = c3d_model.inference_c3d(
                    images_placeholder[gpu_index * batch_size:(gpu_index + 1) * batch_size,:,:,:,:],
                    0.5,
                    batch_size,
                    weights,
                    biases
                )
                loss_name_scope = ('gpud_%d_loss' % gpu_index)
                loss = tower_loss(
                    loss_name_scope,
                    logit,
                    labels_placeholder[gpu_index * batch_size:(gpu_index + 1) * batch_size]
                )
                # Applied only to all variables (fine tuning)
                grads1 = opt_stable.compute_gradients(loss, varlist1)
                # Applied only to out variables
                grads2 = opt_finetuning.compute_gradients(loss, varlist2)
                tower_grads1.append(grads1)
                tower_grads2.append(grads2)
                logits.append(logit)

        logits = tf.concat(logits,0)
        accuracy = tower_acc(logits, labels_placeholder)
        grads1 = average_gradients(tower_grads1)
        grads2 = average_gradients(tower_grads2)
        apply_gradient_op1 = opt_stable.apply_gradients(grads1)
        apply_gradient_op2 = opt_finetuning.apply_gradients(grads2, global_step=global_step)
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Minimizer for transfer learning and finetuning
        last_layer_train_op = tf.group(apply_gradient_op2, variables_averages_op)
        full_train_op = tf.group(apply_gradient_op1, apply_gradient_op2, variables_averages_op)

        # Restore all the layers excluding the last one
        exclude_variables = ['var_name/wout', 'var_name/bout']
        restore_variables = [v.name for v in tf.trainable_variables(scope='var_name')]
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(include=restore_variables, exclude=exclude_variables)
        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_filename, variables_to_restore)

        # Create a saver for writing training checkpoints.
        saver_variables = tf.trainable_variables(scope='var_name')
        saver = tf.train.Saver(saver_variables)

        # Create a session for running Ops on the Graph.
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        # Create summary writter
        train_writer = tf.summary.FileWriter('./visual_logs/train', sess.graph)
        test_writer = tf.summary.FileWriter('./visual_logs/test', sess.graph)

        # Path to train and test tfrecords
        train_data_path = 'tfrecords/train_0.tfrecords'
        val_data_path = 'tfrecords/test_0.tfrecords'

        # Dataset
        filename = tf.placeholder(tf.string, shape=[None])
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(generate_tfrecords._parse_function)
        dataset = dataset.shuffle(buffer_size=batch_size * 10)
        dataset = dataset.batch(batch_size)

        # Iterator
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        # Tfrecords size
        train_size = sum(1 for _ in tf.python_io.tf_record_iterator(train_data_path))
        val_size = sum(1 for _ in tf.python_io.tf_record_iterator(val_data_path))

        # Initialize all variables
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        # sess.run(tf.global_variables_initializer())

        # Load the pretrained weights
        if use_pretrained_model:
            init_fn(sess)
            # saver.restore(sess, model_filename)

        # Extract from checkpoint filename
        start_step = 1
        try:
            start_step = int(model_filename.split('-')[1]) + 1
        except:
            pass

        # Execute training
        for step in xrange(start_step, epochs + 1):
            if step % evaluate_every == 1 or evaluate_every == 1:
                print("\nTraining")
            try:
                with tqdm(desc="Epoch " + str(step) + "/" + str(epochs), total=train_size, file=sys.stdout) as pbar:
                    sess.run(iterator.initializer, feed_dict={filename: [train_data_path]})
                    while True:
                        train_images, train_labels = sess.run(next_element)

                        # Train last layer and finetunning
                        if step <= int(epochs * 0.3) and use_pretrained_model:
                            sess.run(last_layer_train_op, feed_dict={images_placeholder: train_images,
                                                                     labels_placeholder: train_labels})
                        else:
                            sess.run(full_train_op, feed_dict={images_placeholder: train_images,
                                                               labels_placeholder: train_labels})
                        pbar.update(len(train_labels))

            except tf.errors.OutOfRangeError:
                pass

            # Save a checkpoint and evaluate the model periodically.
            if step % evaluate_every == 0 or (step + 1) == epochs or step ==1:
                saver.save(sess, os.path.join(model_save_dir, 'model'), global_step=step)
                acc_train = []
                acc_val = []

                print("\nAccuracy")

                # Training
                try:
                    with tqdm(desc="Epoch " + str(step) + "/" + str(epochs), total=train_size, file=sys.stdout) as pbar:
                        sess.run(iterator.initializer, feed_dict={filename: [train_data_path]})
                        while True:
                            train_images, train_labels = sess.run(next_element)

                            acc = sess.run(
                                accuracy, feed_dict={images_placeholder: train_images, labels_placeholder: train_labels})
                            acc_train.append(acc)
                            pbar.update(len(train_labels))
                except tf.errors.OutOfRangeError:
                    acc_total = np.mean(acc_train)
                    summary = tf.Summary()
                    summary.value.add(tag="Training Accuracy", simple_value=acc_total)
                    train_writer.add_summary(summary, step)
                    print("\tTraining: " + "{:.5f}".format(acc_total))

                # Testing
                try:
                    with tqdm(desc="Epoch " + str(step) + "/" + str(epochs), total=val_size, file=sys.stdout) as pbar:
                        sess.run(iterator.initializer, feed_dict={filename: [val_data_path]})
                        while True:
                            val_images, val_labels = sess.run(next_element)

                            acc = sess.run(
                                accuracy, feed_dict={images_placeholder: val_images, labels_placeholder: val_labels})
                            acc_val.append(acc)
                            pbar.update(len(val_labels))
                except tf.errors.OutOfRangeError:
                    acc_total = np.mean(acc_val)
                    summary = tf.Summary()
                    summary.value.add(tag="Testing Accuracy", simple_value=acc_total)
                    test_writer.add_summary(summary, step)
                    print("\tTesting: " + "{:.5f}".format(acc_total))

        sess.close()

def main(epochs, batch_size, evaluate_every, use_pretrained_model):
    run_training(epochs, batch_size, evaluate_every, use_pretrained_model)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train model for body moviment recognition.')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size')
    parser.add_argument("--evaluate_every", type=int, default=1, help="number of epochs to evaluate model")
    parser.add_argument("--use_pretrained_model", type=lambda x:bool(distutils.util.strtobool(x)), default=True, help="use pretrained model of train from scratch (default: False)")
    args = parser.parse_args()

    main(args.epochs, args.batch_size, args.evaluate_every, args.use_pretrained_model)
