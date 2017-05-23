import mnist_nn
import os
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

data_dir = '/tmp/tensorflow/mnist/input_data'

# Import data
mnist = input_data.read_data_sets(data_dir, one_hot=True)

# Train model
(save_path, model_name) = mnist_nn.train(data_dir)
# save_path = "mnist.model"

with tf.Session() as sess:
  saver = tf.train.import_meta_graph(os.path.join(save_path, model_name + ".meta"))
  saver.restore(sess, os.path.join(save_path, model_name))

  preds = np.argmax(sess.run('softmax_logits:0', feed_dict={'x:0':mnist.test.images, 'keep_prob:0':1.0}), axis=1)
  labels = np.argmax(mnist.test.labels, axis=1)

  accuracy = float(np.sum(preds == labels))/float(len(preds))
  print("Accuracy:\n%g" % accuracy)
