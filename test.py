import mnist_softmax

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import os
import numpy as np

data_dir = '/tmp/tensorflow/mnist/input_data'

# Import data
mnist = input_data.read_data_sets(data_dir, one_hot=True)

# Train model
(save_path, model_name) = mnist_softmax.train(data_dir)


with tf.Session() as sess:
  saver = tf.train.import_meta_graph(os.path.join(save_path, model_name + ".meta"))
  saver.restore(sess, os.path.join(save_path, model_name))

  preds = np.argmax(sess.run('prediction:0', feed_dict={'x:0':mnist.test.images}), axis=1)
  labels = np.argmax(mnist.test.labels, axis=1)

  accuracy = float(np.sum(preds == labels))/float(len(preds))
  print("Accuracy:\n%g" % accuracy)


# # W = tf.Variable(tf.zeros([784, 10]))
# # b = tf.Variable(tf.zeros([10]))
# # saver = tf.train.Saver({'W': W, 'b': b})
# #
# # sess = tf.InteractiveSession()
# # saver.restore(sess, "./mnist.model")
# #
# # x = tf.placeholder(tf.float32, [None, 784])
# # y_ = tf.placeholder(tf.float32, [None, 10])
# # y = tf.matmul(x, W) + b
#
# # Test trained model
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print("Test accuracy is:")
# print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
