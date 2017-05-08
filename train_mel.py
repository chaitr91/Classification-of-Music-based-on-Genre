import pickle
import random

import numpy as np
import sys
import tensorflow as tf
from helper_util import *


if __name__ == "__main__":

    # Parameters
    step_size = 64
    ckpt_step = 1
    learning_rate = 0.001
    training_iters = 50000

    # Below are the network hyper-parameters
    n_input = 599 * 128 * 5
    n_classes = 6
    dropout = 0.75

    # Load data
    train_data = []
    with open(sys.argv[1], 'rb') as f:
        content = f.read()
        train_data = pickle.loads(content)
    train_data = np.asarray([train_data[i] for i in np.asarray([l for l in train_data])])

    train_data = train_data.reshape((train_data.shape[0], n_input))

    train_labels = []
    with open(sys.argv[2], 'rb') as f:
        content = f.read()
        train_labels = pickle.loads(content)
    print(train_labels.shape)

    # Randomize the input data

    shuffle_index = list(range(0,len(train_data)))
    random.shuffle(shuffle_index)

    print(shuffle_index)
    train_data = train_data[shuffle_index]
    train_labels = train_labels[shuffle_index]


    #graph input for tensorflow
    keep_prob = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])



    # Construct the model
    pred_res = convolution_network(x, keep_prob)

    #Adam Optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_res, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # Initializing the variables
    init = tf.initialize_all_variables()

    #This is for save and restore
    saver = tf.train.Saver()

    # Launch graph
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        sess.run(init)
        step = 1
        current_size = step*step_size

        while current_size < training_iters:

            current_labels = []
            current_dataset = []

            batch_start = (current_size) % len(train_data)
            batch_end = (current_size + step_size) % len(train_data)
            print("TRAIN LEN",len(train_data))
            print("start : ",batch_start)
            print("End : ", batch_end)
            if batch_start < batch_end:
                currentX = train_data[batch_start:batch_end]
                currentY = train_labels[batch_start:batch_end]
            else:
                current_dataset = np.vstack((train_data[batch_start:], train_data[:batch_end]))
                current_labels = np.vstack((train_labels[batch_start:], train_labels[:batch_end]))

                currentX =  current_dataset
                currentY =  current_labels
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: currentX, y: currentY, keep_prob: dropout})

            if step % ckpt_step == 0:

                ckpt_path = saver.save(sess, "model.ckpt")
                print("The checkpoint is saved in: %s" % ckpt_path)
            step += 1
            current_size = step * step_size

        model_path = saver.save(sess, "model.final")
        print("The final model is present in: %s" % model_path)
