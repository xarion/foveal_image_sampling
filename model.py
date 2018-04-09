import tensorflow as tf
import numpy as np

from tensorflow.contrib.layers import batch_norm
from tensorflow.python.ops import control_flow_ops as control

from clutter_it_mnist import clutter_it


class Model:
    def __init__(self):
        #  default mode is training
        self.mode = tf.placeholder_with_default(tf.constant(1, dtype=tf.int32), shape=[])

        self.x_mus, self.y_mus, self.sigmas = self.get_kernel_parameters()
        self.kernels = self.get_kernels()
        self.mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        self.images, self.labels = self.get_data_placeholders()
        self.loss, self.training_op, self.predictions, self.accuracy, self.global_step = self.define_model()

    def get_kernel_parameters(self):
        x_mus = tf.range(start=10, limit=90, delta=80./12.)
        y_mus = tf.range(start=10, limit=90, delta=80./12.)

        x_mus, y_mus = tf.meshgrid(x_mus, y_mus)

        # accept those as the initial values.
        x_mus = tf.Variable(x_mus)
        y_mus = tf.Variable(y_mus)
        sigmas = tf.Variable(tf.ones([12, 12]) * 0.07)
        return x_mus, y_mus, sigmas

    def get_kernels(self):
        x_dists = tf.distributions.Normal(loc=self.x_mus, scale=self.sigmas)
        y_dists = tf.distributions.Normal(loc=self.y_mus, scale=self.sigmas)

        locs = tf.range(start=0, limit=100, delta=1.)
        x_locs, y_locs = tf.meshgrid(locs, locs)

        x_locs = tf.tile(tf.reshape(x_locs, [100, 100, 1, 1]), [1, 1, 12, 12])
        y_locs = tf.tile(tf.reshape(y_locs, [100, 100, 1, 1]), [1, 1, 12, 12])

        x_values = x_dists.prob(x_locs)
        y_values = y_dists.prob(y_locs)

        kernels = x_values * y_values
        return tf.reshape(kernels, [100 * 100, 12 * 12])

    def get_numpy_data(self, mode, batch_size=100):
        if mode is 1:
            dataset = self.mnist.train  # Returns np.array
            subsamples = np.random.randint(0, dataset.num_examples, batch_size)
        elif mode is 2:
            dataset = self.mnist.validation
            subsamples = np.random.randint(0, dataset.num_examples, batch_size)
        else:
            dataset = self.mnist.test
            batch_size = dataset.num_examples
            subsamples = np.arange(0, batch_size)

        data = dataset.images[subsamples]  # Returns np.array
        labels = np.asarray(dataset.labels[subsamples], dtype=np.int32)

        cluttered_data = clutter_it(data, self.mnist, flag_c=mode, flag_d=1)

        return cluttered_data, labels

    def get_data_placeholders(self, batch_size=100):
        return tf.placeholder(tf.float32, [batch_size, 100 * 100]), tf.placeholder(tf.int32, [batch_size])
        # return images, labels

    def define_model(self):
        with tf.variable_scope("kernels"):
            l = tf.matmul(self.images, self.kernels)
            l = batch_norm(l, fused=True)
            l = tf.nn.relu(l)

        with tf.variable_scope("fc_1"):
            weights = tf.Variable(tf.random_normal([144, 512], stddev=0.1), name="weights")
            l = tf.matmul(l, weights)
            l = batch_norm(l, fused=True)
            l = tf.nn.relu(l)

        with tf.variable_scope("fc_2"):
            weights = tf.Variable(tf.random_normal([512, 512], stddev=0.1), name="weights")
            l = tf.matmul(l, weights)
            l = batch_norm(l, fused=True)
            l = tf.nn.relu(l)

        with tf.variable_scope("fc_3"):
            weights = tf.Variable(tf.random_normal([512, 512], stddev=0.1), name="weights")
            l = tf.matmul(l, weights)
            l = batch_norm(l, fused=True)
            l = tf.nn.relu(l)

        with tf.variable_scope("fc_4"):
            weights = tf.Variable(tf.random_normal([512, 10], stddev=0.1), name="weights")
            l = tf.matmul(l, weights)
            bias = tf.Variable(tf.zeros([10]))
            l = l + bias

        with tf.variable_scope("predictions"):
            predictions = tf.cast(tf.argmax(l, axis=1), tf.int32)

        with tf.variable_scope("accuracy"):
            correct_prediction = tf.equal(predictions, self.labels)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

        with tf.variable_scope("loss"):
            loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=l)
            tf.summary.scalar('loss', loss)

        with tf.variable_scope("training"):
            with tf.device("/cpu:0"):
                global_step = tf.Variable(0, name='global_step', trainable=False)
                boundaries = [45000, 57000]
                values = [0.1, 0.01, 0.001]
                learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
                tf.summary.scalar('learning_rate', learning_rate)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            boundaries = [5000, 7000]
            values = [0.1, 0.01, 0.001]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
            tf.summary.scalar('learning_rate', learning_rate)

            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                training_op = optimizer.minimize(loss, global_step=global_step)

        return loss, training_op, predictions, accuracy, global_step
