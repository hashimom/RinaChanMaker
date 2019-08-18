import tensorflow as tf


class Generator:
    def __init__(self, x_size, y_size, z_dim):
        self.x_size = x_size
        self.y_size = y_size
        self.z_dim = z_dim

        # all-1
        self.W_fc1 = tf.Variable(tf.random_normal([z_dim, 1028], stddev=0.02))
        self.b_fc1 = tf.Variable(tf.zeros([1028]))

        # all-2
        self.W_fc2 = tf.Variable(tf.random_normal([1028, (5 * 5 * 64)], stddev=0.02))
        self.b_fc2 = tf.Variable(tf.zeros([(5 * 5 * 64)]))

        # conv2d-1
        self.W_conv1 = tf.Variable(tf.random_normal([9, 9, 32, 64], stddev=0.02))

        # conv2d-2
        self.W_conv2 = tf.Variable(tf.random_normal([18, 18, 16, 32], stddev=0.02))

        # conv2d-3
        self.W_conv3 = tf.Variable(tf.random_normal([x_size, y_size, 3, 16], stddev=0.02))

    def var_list(self):
        return [self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2, self.W_conv1, self.W_conv2, self.W_conv3]

    def __call__(self, batch_size=1):
        """

        :return:
        """
        inputs = tf.random_uniform([batch_size, self.z_dim], minval=-1.0, maxval=1.0)

        h_fc1 = tf.matmul(inputs, self.W_fc1) + self.b_fc1
        h_fc1 = tf.nn.leaky_relu(self.batch_norm(h_fc1, [0, 1]))

        h_fc2 = tf.matmul(h_fc1, self.W_fc2) + self.b_fc2
        h_fc2 = tf.nn.leaky_relu(self.batch_norm(h_fc2, [0, 1]))
        h_fc2 = tf.nn.dropout(h_fc2, 0.5)
        h_conv_in = tf.reshape(h_fc2, [-1, 5, 5, 64])

        # 畳み込み層1
        h_conv1 = self.conv2d_transpose(h_conv_in, self.W_conv1, [9, 9, 32])
        h_conv1 = tf.nn.leaky_relu(self.batch_norm(h_conv1, [0, 1, 2]))

        # 畳み込み層1
        h_conv2 = self.conv2d_transpose(h_conv1, self.W_conv2, [18, 18, 16])
        h_conv2 = tf.nn.leaky_relu(self.batch_norm(h_conv2, [0, 1, 2]))

        # 畳み込み層3
        img = self.conv2d_transpose(h_conv2, self.W_conv3, [self.x_size, self.y_size, 3])
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


class Discriminator:
    def __init__(self, x_size, y_size):
        """ コンストラクタ

        """
        self.x_size = x_size
        self.y_size = y_size

        # conv2d-1
        self.W_conv1 = tf.Variable(tf.truncated_normal([4, 4, 3, 32], stddev=0.1))
        self.b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

        # conv2d-2
        self.W_conv2 = tf.Variable(tf.truncated_normal([9, 9, 32, 64], stddev=0.1))
        self.b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

        # all-1
        self.W_fc1 = tf.Variable(tf.truncated_normal([(9 * 9 * 64), 1024], stddev=0.1))
        self.b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

        # all-2
        self.W_fc2 = tf.Variable(tf.truncated_normal([1024, 2], stddev=0.1))
        self.b_fc2 = tf.Variable(tf.constant(0.1, shape=[2]))

    def var_list(self):
        return [self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2]

    def __call__(self, images_placeholder, y_labels, keep_prob):
        """ モデル作成

        :param images_placeholder:
        :param labels:
        :param keep_prob:
        :return:
        """
        # 画像を行列に変換
        x_image = tf.reshape(images_placeholder, [-1, self.x_size, self.y_size, 3])

        # 畳み込み層1
        h_conv1 = tf.nn.relu(self.conv2d(x_image, self.W_conv1) + self.b_conv1)

        # プーリング層1
        h_pool1 = self.max_pool_2x2(h_conv1)

        # 畳み込み層2
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, self.W_conv2) + self.b_conv2)

        # プーリング層2
        h_pool2 = self.max_pool_2x2(h_conv2)

        # 全結合層1
        h_pool2_flat = tf.reshape(h_pool2, [-1, 9*9*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1)
        h_fc1 = tf.nn.dropout(h_fc1, keep_prob)

        # 特徴量→各ラベルの確立へ変換
        y = tf.matmul(h_fc1, self.W_fc2) + self.b_fc2
        loss = tf.losses.softmax_cross_entropy(onehot_labels=y_labels, logits=y)
        return loss

    @staticmethod
    def conv2d(x, W):
        """ 畳み込み層

        :param x:
        :param W:
        :return:
        """
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        """ プーリング層

        :param x:
        :return:
        """
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
