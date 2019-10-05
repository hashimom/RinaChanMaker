import tensorflow as tf


class Generator(tf.Module):
    def __init__(self, x_size, y_size, z_dim):
        self.x_size = x_size
        self.y_size = y_size
        self.z_dim = z_dim
        init_w = tf.random_normal_initializer()
        init_b = tf.zeros_initializer()

        # all-1
        self.W_fc1 = tf.Variable(init_w(shape=(z_dim, (16 * 16 * 256)), dtype="float32"))
        self.b_fc1 = tf.Variable(init_b(shape=(16 * 16 * 256)))

        # conv2d-1
        self.W_conv1 = tf.Variable(init_w(shape=(32, 32, 128, 256), dtype="float32"))

        # conv2d-2
        self.W_conv2 = tf.Variable(init_w(shape=(64, 64, 64, 128), dtype="float32"))

        # conv2d-3
        self.W_conv3 = tf.Variable(init_w(shape=(128, 128, 32, 64), dtype="float32"))

        # conv2d-4
        self.W_conv4 = tf.Variable(init_w(shape=(256, 256, 16, 32), dtype="float32"))

        # conv2d-5
        self.W_conv5 = tf.Variable(init_w(shape=(x_size, y_size, 3, 16), dtype="float32"))


    @tf.function
    def __call__(self, batch_size=1):
        """

        :return:
        """
        inputs = tf.random.uniform([batch_size, self.z_dim], minval=-1.0, maxval=1.0)

        h_fc1 = tf.matmul(inputs, self.W_fc1) + self.b_fc1
        h_fc1 = tf.nn.leaky_relu(self.batch_norm(h_fc1, [0, 1]))
        h_conv_in = tf.reshape(h_fc1, [-1, 16, 16, 256])

        # 畳み込み層1
        h_conv = self.conv2d_transpose(h_conv_in, self.W_conv1, [32, 32, 128])
        h_conv = tf.nn.leaky_relu(self.batch_norm(h_conv, [0, 1, 2]))

        # 畳み込み層1
        h_conv = self.conv2d_transpose(h_conv, self.W_conv2, [64, 64, 64])
        h_conv = tf.nn.leaky_relu(self.batch_norm(h_conv, [0, 1, 2]))

        # 畳み込み層1
        h_conv = self.conv2d_transpose(h_conv, self.W_conv3, [128, 128, 32])
        h_conv = tf.nn.leaky_relu(self.batch_norm(h_conv, [0, 1, 2]))

        # 畳み込み層1
        h_conv = self.conv2d_transpose(h_conv, self.W_conv4, [256, 256, 16])
        h_conv = tf.nn.leaky_relu(self.batch_norm(h_conv, [0, 1, 2]))

        # 畳み込み層5
        img = self.conv2d_transpose(h_conv, self.W_conv5, [self.x_size, self.y_size, 3])
        y_conv = tf.reshape(tf.math.tanh(img), [-1, (self.x_size * self.y_size * 3)])
        return y_conv

    @staticmethod
    def batch_norm(x, axes):
        mean, var = tf.nn.moments(x, axes)
        with tf.control_dependencies([mean, var]):
            return tf.nn.batch_normalization(x, mean, var, None, None, 1e-10)

    @staticmethod
    def conv2d_transpose(x, W, shape):
        """ 畳み込み層

        :param x:
        :param W:
        :return:
        """
        output_shape = [tf.shape(x)[0], shape[0], shape[1], shape[2]]
        return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1, 2, 2, 1], padding='SAME')

