import tensorflow as tf


class Discriminator(tf.Module):
    def __init__(self, x_size, y_size):
        """ コンストラクタ

        """
        self.x_size = x_size
        self.y_size = y_size
        init_w = tf.random_normal_initializer()
        init_b = tf.zeros_initializer()

        # conv2d-1
        self.W_conv1 = tf.Variable(init_w(shape=(4, 4, 3, 32), dtype="float32"))
        self.b_conv1 = tf.Variable(init_b(shape=32))

        # conv2d-2
        self.W_conv2 = tf.Variable(init_w(shape=(9, 9, 32, 64), dtype="float32"))
        self.b_conv2 = tf.Variable(init_b(shape=64))

        # all-1
        self.W_fc1 = tf.Variable(init_w(shape=((9 * 9 * 64), 1024), dtype="float32"))
        self.b_fc1 = tf.Variable(init_b(shape=1024))

        # all-2
        self.W_fc2 = tf.Variable(init_w(shape=(1024, 2),  dtype="float32"))
        self.b_fc2 = tf.Variable(init_b(shape=2))

    @tf.function
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
        logits = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y_labels, depth=2), logits=y)
        loss = tf.reduce_mean(logits)
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
