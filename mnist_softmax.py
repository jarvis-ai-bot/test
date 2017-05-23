# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division

import argparse
import sys
import os
import json

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

def save_model_for_clipper(sess, model_save_dir=None, model_name=None, input_node_name=None, output_node_name=None, placeholder_vars={}):
    with sess.graph.as_default():
      saver = tf.train.Saver(sharded=False)
      saver.save(sess, os.path.join(model_save_dir, model_name), meta_graph_suffix='meta', write_meta_graph=True)
    config_dict = {
        "model_name": model_name,
        "input_node_name": input_node_name,
        "output_node_name": output_node_name,
        "placeholder_vars": placeholder_vars
    }
    with open(os.path.join(model_save_dir, "model_config.json"), "w") as config_file:
      json.dump(config_dict, config_file)

def train(data_dir):
  # Import data
  mnist = input_data.read_data_sets(data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784], name="x")
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.nn.xw_plus_b(x, W, b, name="prediction")
  # y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  saver = tf.train.Saver({'W': W, 'b': b})

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  model_save_dir = os.path.abspath(".")
  model_name = "mnist.model"
  save_model_for_clipper(
      sess,
      model_save_dir=model_save_dir,
      model_name=model_name,
      input_node_name="x",
      output_node_name="prediction")
  return (model_save_dir, model_name)

  # return saver.save(sess, './mnist.model')
