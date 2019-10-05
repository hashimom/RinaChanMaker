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
        self.W_conv2 = tf.Variable(init_w(shape=(8, 8, 32, 64), dtype="float32"))
        self.b_conv2 = tf.Variable(init_b(shape=64))

        # conv2d-3
        self.W_conv3 = tf.Variable(init_w(shape=(16, 16, 64, 128), dtype="float32"))
        self.b_conv3 = tf.Variable(init_b(shape=128))

        # conv2d-4
        self.W_conv4 = tf.Variable(init_w(shape=(32, 32, 128, 256), dtype="float32"))
        self.b_conv4 = tf.Variable(init_b(shape=256))

        # all
        self.W_fc1 = tf.Variable(init_w(shape=((32 * 32 * 256), 2), dtype="float32"))
        self.b_fc1 = tf.Variable(init_b(shape=2))

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

        # 畳み込み層 +  プーリング層 1
        h_conv = tf.nn.leaky_relu(self.conv2d(x_image, self.W_conv1) + self.b_conv1)
        h_conv = self.max_pool_2x2(h_conv)

        # 畳み込み層 +  プーリング層 2
        h_conv = tf.nn.leaky_relu(self.conv2d(h_conv, self.W_conv2) + self.b_conv2)
        h_conv = self.max_pool_2x2(h_conv)

        # 畳み込み層 +  プーリング層 3
        h_conv = tf.nn.leaky_relu(self.conv2d(h_conv, self.W_conv3) + self.b_conv4)
        h_conv = self.max_pool_2x2(h_conv)

        # 畳み込み層 +  プーリング層 4
        h_conv = tf.nn.leaky_relu(self.conv2d(h_conv, self.W_conv3) + self.b_conv4)
        h_conv = self.max_pool_2x2(h_conv)

        # 全結合層
        h_conv = tf.reshape(h_conv, [-1, 32*32*256])
        h_conv = tf.nn.leaky_relu(tf.matmul(h_conv, self.W_fc1) + self.b_fc1)
        y = tf.nn.dropout(h_conv, keep_prob)

        # 特徴量→各ラベルの確立へ変換
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
