import tensorflow as tf
import numpy as np
import pickle
import sys
from helper_util import *

n_input = 599 * 128*5
n_classes = 6
dropout = 0.75
learning_rate = 0.01
# Load data
data = []
with open(sys.argv[1], 'rb') as f:
    content = f.read()
    data = pickle.loads(content)
data = np.asarray([data[i] for i in np.asarray([l for l in data])])

data = data.reshape((data.shape[0], n_input))

labels = []
with open(sys.argv[2], 'rb') as f:
    content = f.read()
    labels = pickle.loads(content)
print(labels.shape)



testData = data
testLabels = labels

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)




# Construct model
pred = convolution_network(x, keep_prob)
 
# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Needs model.final in the current directory
ckpt = tf.train.get_checkpoint_state("./")
 
# Launch the graph
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    sess.run(init)
    saver.restore(sess, ckpt.model_checkpoint_path)
    predictions = sess.run(accuracy, feed_dict={x: testData, y:testLabels, keep_prob:1. })
    print(predictions)