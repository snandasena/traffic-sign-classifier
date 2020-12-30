# required imports
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

with open('./trainset.p', mode='rb') as f:
    trainset = pickle.load(f)

X_train = trainset['train_x']
y_train = trainset['train_y']

with open('./testtset.p', mode='rb') as f:
    testset = pickle.load(f)

X_test = testset['test_x']
y_test = testset['test_y']

with open('./valid.p', mode='rb') as f:
    valid = pickle.load(f)

X_valid = valid['valid_x']
y_valid = valid['valid_y']

# Training params

n_classes = 43
l_rate = 0.002
epoches = 65
batch_size = 130
display_steps = 50
flatten = layers.Flatten()
dropout = layers.Dropout(rate=0.53)
mu = 0.0
sigma = 0.1


# Create a some wrapper
def conv2d(x, W, b, stride=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)

    return tf.nn.relu(x)


def maxpoll2d(x, k=2):
    # Maxpool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')


# Hyperparameters

random_normal = tf.initializers.RandomNormal(mean=mu, stddev=sigma)

weights = {
    # Conv layer 1: 5x5 conv, 1 input, 32 filters
    'wc1': tf.Variable(random_normal([5, 5, 1, 6])),
    # Conv layer 2: 5x5 conv, 32 inouts, 64 filters
    'wc2': tf.Variable(random_normal([5, 5, 6, 16])),
    # Fully connected layer: 7x7x64 inputs, 1024 units
    'wd1': tf.Variable(random_normal([5, 5, 16, 400])),
    # FC output layer: 1024 inputs, 43 units
    'out': tf.Variable(random_normal([800, 43]))
}

biases = {
    'bc1': tf.Variable(tf.zeros(6)),
    'bc2': tf.Variable(tf.zeros(16)),
    'bd1': tf.Variable(tf.zeros(400)),
    'out': tf.Variable(tf.zeros(43))
}


def conv_net(x):
    # Convolution layer. Output shape: [-1, 28, 28, 32]
    x = tf.reshape(x, [-1, 32, 32, 1])

    x = conv2d(x, weights['wc1'], biases['bc1'])

    # Max pooling(down sampling). output shape [-1, 14,14,32]
    x = maxpoll2d(x, k=2)
    # Convolution layer: Output shape: [-1, 14,14, 64]
    x = conv2d(x, weights['wc2'], biases['bc2'])

    # Max pooling. output shape [-1, 7, 7, 64]
    x = maxpoll2d(x, k=2)
    layer2 = x

    x = conv2d(x, weights['wd1'], biases['bd1'])

    l2_flat = flatten(layer2)
    x_flat = flatten(x)

    res = tf.concat(values=[l2_flat, x_flat], axis=1)

    x = dropout(res, training=True)

    out = tf.add(tf.matmul(x, weights['out']), biases['out'])

    return tf.nn.softmax(out)


optimizer = tf.optimizers.Adam(l_rate)


# Cross-entropy
def cross_entropy(y_pred, y_true):
    # Encode labels to a one hot vector
    y_true = tf.one_hot(y_true, depth=n_classes)

    # Clip prediction values to avoid log(0) error
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)

    # Compute cross-entropy
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))


# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


# Optimization process
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automate differentiation

    with tf.GradientTape() as g:
        pred = conv_net(x)
        loss = cross_entropy(pred, y)

    # Variables to update, i.e trainable vars
    trainable_vars = list(weights.values()) + list(biases.values())

    # Compute gradients
    gradients = g.gradient(loss, trainable_vars)

    # Update W and b following gradient
    optimizer.apply_gradients(zip(gradients, trainable_vars))


# Shuffle data using tensorflow tf.data
X_train, X_test, X_valid = np.array(X_train, np.float32), np.array(X_test, np.float32), np.array(X_valid, np.float32)
# train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)

from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
n = len(X_train)
step = 0
for i in range(epoches):
    X_train, y_train = shuffle(X_train, y_train)
    for offset in range(0, n, batch_size):
        step += 1
        end = offset + batch_size
        batch_x, batch_y = X_train[offset:end], y_train[offset:end]
        run_optimization(batch_x, batch_y)

        if step % display_steps == 0:
            pred = conv_net(batch_x)
            loss = cross_entropy(pred, batch_y)
            acc = accuracy(pred, batch_y)

            print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

# Test  model on validation set
pred1 = conv_net(X_validation)
print("Train Validatation accuracy: %f" % accuracy(pred1, y_validation))

# Test  model on validation set
pred = conv_net(X_test)
print("Test accuracy: %f" % accuracy(pred, y_test))

pred2 = conv_net(X_valid)
print("Valid accuracy: %f" % accuracy(pred2, y_valid))
