import tensorflow as tf

number_of_kernels = 144

x_mus = tf.range(start=25, limit=75, delta=4.54545)
y_mus = tf.range(start=25, limit=75, delta=4.54545)

x_mus, y_mus = tf.meshgrid(x_mus, y_mus)

# accept those as the initial values.
x_mus = tf.Variable(x_mus)
y_mus = tf.Variable(y_mus)
sigmas = tf.Variable(tf.ones([12, 12]) * 1.7)

x_dists = tf.distributions.Normal(loc=x_mus, scale=sigmas)
y_dists = tf.distributions.Normal(loc=y_mus, scale=sigmas)

locs = tf.range(start=0, limit=100, delta=1.)
x_locs, y_locs = tf.meshgrid(locs, locs)

x_locs = tf.tile(tf.reshape(x_locs, [100, 100, 1, 1]), [1, 1, 12, 12])
y_locs = tf.tile(tf.reshape(y_locs, [100, 100, 1, 1]), [1, 1, 12, 12])

x_values = x_dists.prob(x_locs)
y_values = y_dists.prob(y_locs)

kernels = x_values * y_values


input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 100, 100])

kernels = tf.reshape(kernels, [100 * 100, 12 * 12])
reshaped_input = tf.reshape(input_placeholder, [None, 100*100])

l_1 = tf.matmul(reshaped_input, kernels)
biases = tf.zeros([12*12])
l_1 = tf.nn.relu(l_1 + biases)
