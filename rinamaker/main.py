import tensorflow as tf
import numpy as np
import os
import glob
import shutil
import cv2
from rinamaker.model import Generator, Discriminator

# 教師データサイズ
image_x = 36
image_y = 36
image_size = image_x * image_y
# RGB
image_data_size = image_size * 3

batch_size = 512
Z_dim = 100
epoch_num = 50000

# 画像生成用ディレクトリの作成
if not os.path.isdir('../output'):
    os.makedirs('../output')


def read_image(base_path):
    """ 画像リード

    :param base_path: 画像ベースパス
    :return:
    """
    # 画像変換
    image_ary = []
    file_list = glob.glob(os.path.abspath(base_path + "/*.png"))
    for file in file_list:
        img = cv2.imread(file)
        img = cv2.resize(img, (image_x, image_y))
        image_ary.append(img.flatten().astype(np.float32) / 255.0)

    # numpy形式に変換
    # return np.asarray(image_ary)
    return image_ary


def g_train(gen_image, disc, y_labels):
    loss = disc(gen_image, y_labels, 1.0)
    return loss


def d_train(disc, img, y_labels):
    loss = disc(img, y_labels, 0.5)
    return loss

# 画像読み込み
train_data = read_image("../train/")
batch_size = len(train_data)

D_real_labels = np.ones(batch_size)
D_fake_labels = np.zeros(batch_size)
G_labels = np.ones(batch_size)

# placeholder
Z = tf.placeholder(tf.float32, shape=[None, Z_dim], name='Z')
images = tf.placeholder(tf.float32, shape=[None, image_data_size], name='X')
labels = tf.placeholder(tf.uint8, shape=[None], name='Y')
Y = tf.one_hot(labels, depth=2, dtype=tf.float32)

l_rate = 0.00001
learning_rate = tf.placeholder(tf.float32)

# Model
generator = Generator(image_x, image_y, Z_dim)
discriminator = Discriminator(image_x, image_y)

# Setup
D_loss = d_train(discriminator, images, Y)
D_solver = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(D_loss, var_list=discriminator.var_list())

G = generator(batch_size)
G_loss = g_train(G, discriminator, Y)
G_solver = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(G_loss, var_list=generator.var_list())

cwd = os.getcwd()
model_path = cwd + "/model/"
if os.path.isdir(model_path):
    shutil.rmtree(model_path)

saver = tf.train.Saver()
session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
with tf.Session(config = session_config) as sess:
    sess.run(tf.global_variables_initializer())
    # 学習プロセス開始
    for itr in range(epoch_num):
        if itr % 1000 == 0:
            try:
                dirname = "../output/%06d/" % itr
                if not os.path.isdir(dirname):
                    os.makedirs(dirname)
                for i in range(9):
                    gene_image = generator().eval()
                    img_obj = gene_image.reshape(image_x, image_y, 3) * 255.0
                    file_name = dirname + "%02d.png" % i
                    cv2.imwrite(file_name, img_obj)
            except:
                pass

            # 学習率の更新
            l_rate *= 1.1
            print("Learning rate Change!! -> %.6f" % l_rate)

        # Discriminator-Real
        rand_data = []
        rand_idx = np.random.randint(0, len(train_data), batch_size)
        for i in rand_idx:
            # ノイズを追加してランダムに並び替え
            gauss = np.random.normal(0, 0.005, train_data[i].shape)
            train_image_gs = train_data[i] + gauss
            rand_data.append(train_data[i])
        train_np_data = np.asarray(rand_data)
        sess.run([D_solver, D_loss], feed_dict={images: train_np_data, labels: D_real_labels, learning_rate: l_rate})

        # Generator & Discriminator-Fake
        for i in range(2):
            _, gene_image, G_loss_curr = sess.run([G_solver, G, G_loss], feed_dict={labels: G_labels, learning_rate: l_rate})
            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={images: gene_image, labels: D_fake_labels, learning_rate: l_rate})

        if itr % 100 == 0:
            print('Iter: {}'.format(itr))
            print(" D loss: " + str(D_loss_curr))
            print(" G_loss: " + str(G_loss_curr))

    # 学習モデル出力
    save_path = saver.save(sess, model_path)
    print("model save!!")

