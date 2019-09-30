import tensorflow as tf


class Generator(tf.Module):
    def __init__(self, x_size, y_size, z_dim):
        self.x_size = x_size
        self.y_size = y_size
        self.z_dim = z_dim
        init_w = tf.random_normal_initializer()
        init_b = tf.zeros_initializer()

        # all-1
        self.W_fc1 = tf.Variable(init_w(shape=(z_dim, 1028), dtype="float32"))
        self.b_fc1 = tf.Variable(init_b(shape=1028))

        # all-2
        self.W_fc2 = tf.Variable(init_w(shape=(1028, (9 * 9 * 64)), dtype="float32"))
        self.b_fc2 = tf.Variable(init_b(shape=(9 * 9 * 64)))

        # conv2d-1s
        self.W_conv1 = tf.Variable(init_w(shape=(18, 18, 32, 64), dtype="float32"))

        # conv2d-2
        self.W_conv2 = tf.Variable(init_w(shape=(x_size, y_size, 3, 32), dtype="float32"))

    @tf.function
    def __call__(self, batch_size=1):
        """

        :return:
        """
        inputs = tf.random.uniform([batch_size, self.z_dim], minval=-1.0, maxval=1.0)

        h_fc1 = tf.matmul(inputs, self.W_fc1) + self.b_fc1
        h_fc1 = tf.nn.leaky_relu(self.batch_norm(h_fc1, [0, 1]))

        h_fc2 = tf.matmul(h_fc1, self.W_fc2) + self.b_fc2
        h_fc2 = tf.nn.leaky_relu(self.batch_norm(h_fc2, [0, 1]))
        h_fc2 = tf.nn.dropout(h_fc2, 0.5)
        h_conv_in = tf.reshape(h_fc2, [-1, 9, 9, 64])

        # 畳み込み層1
        h_conv1 = self.conv2d_transpose(h_conv_in, self.W_conv1, [18, 18, 32])
        h_conv1 = tf.nn.leaky_relu(self.batch_norm(h_conv1, [0, 1]))

        # 畳み込み層2
        img = self.conv2d_transpose(h_conv1, self.W_conv2, [self.x_size, self.y_size, 3])
        y_conv = tf.reshape(tf.math.tanh(img), [-1, (self.x_size * self.y_size * 3)])
        return y_conv

    @staticmethod
    def batch_norm(x, axes):
        mean, var = tf.nn.moments(x, axes)
        with tf.control_dependencies([mean, var]):
            return tf.nn.batch_normalization(x, mean, var, None, None, 1e-5)

    @staticmethod
    def conv2d_transpose(x, W, shape):
        """ 畳み込み層

        :param x:
        :param W:
        :return:
        """
        output_shape = [tf.shape(x)[0], shape[0], shape[1], shape[2]]
        return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1, 2, 2, 1], padding='SAME')

