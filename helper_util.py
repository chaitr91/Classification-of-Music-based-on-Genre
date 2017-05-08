import tensorflow as tf

n_classes = 6

weights = {

        'wc1': tf.Variable(tf.random_normal([4, 4, 5, 149])),
        'wc2': tf.Variable(tf.random_normal([4, 4, 149, 73])),
        'wc3': tf.Variable(tf.random_normal([4, 4, 73, 35])),
        'wd1': tf.Variable(tf.random_normal([38 * 8 * 35, 8192])),
        'out': tf.Variable(tf.random_normal([8192, n_classes]))
    }

bias = {
        'bc1': tf.Variable(tf.random_normal([149])),
        'bc2': tf.Variable(tf.random_normal([73])),
        'bc3': tf.Variable(tf.random_normal([35])),
        'bd1': tf.Variable(tf.random_normal([8192])),
        'out': tf.Variable(tf.random_normal([n_classes]))
}


# conv2d method creates the model
def conv2d(weight, bias, sound_info):
    conv2d_res = tf.nn.conv2d(sound_info, weight, strides=[1, 1, 1, 1], padding='SAME')
    bias_add_res = tf.nn.bias_add(conv2d_res, bias)
    relu_res = tf.nn.relu(bias_add_res)

    return relu_res


def convolution_network(X, dropout):
    # Reshape the input
    X = tf.reshape(X, shape=[-1, 599, 128, 5])

    # Convolution Layer
    conv1 = conv2d(weights['wc1'], bias['bc1'], X)

    # Max Pooling (down-sampling)
    # Using ksize 4
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    # Dropout
    conv1 = tf.nn.dropout(conv1, dropout)

    # Convolution Layer
    conv2 = conv2d(weights['wc2'], bias['bc2'], conv1)

    # Max Pooling (down-sampling)
    # Using ksize 2
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Dropout
    conv2 = tf.nn.dropout(conv2, dropout)

    # Convolution Layer
    conv3 = conv2d(weights['wc3'], bias['bc3'], conv2)

    # Max Pooling (down-sampling)
    # Using ksize =2
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Dropout
    conv3 = tf.nn.dropout(conv3, dropout)

    # Fully connected layer
    dense1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    # Perform Relu activation
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, weights['wd1']),bias['bd1']))

    # Dropout
    dense1 = tf.nn.dropout(dense1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(dense1, weights['out']), bias['out'])

    return out