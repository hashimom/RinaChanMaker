import argparse
import tensorflow as tf
import numpy as np
import os
import glob
import shutil
import cv2
from rinamaker.generator import Generator
from rinamaker.discriminator import Discriminator

# 教師データサイズ
image_x = 36
image_y = 36
image_size = image_x * image_y
# RGB
image_data_size = image_size * 3

d_batch_size = 512
g_batch_size = 512
Z_dim = 100
epoch_num = 20000

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)

class Trainer():
    def __init__(self, target_path, output_path, out_model_path, dummy_path=None):
        # Model
        self.gene = Generator(image_x, image_y, Z_dim)
        self.disc = Discriminator(image_x, image_y)

        # Optimizer
        self.gene_opt = tf.optimizers.Adam(0.00001)
        self.disc_opt = tf.optimizers.Adam(0.00001)

        # 画像生成用ディレクトリの作成
        self.output_path = output_path

        # ターゲット画像の取得
        self.target_image = self.read_image(target_path)
        self.target_len = len(self.target_image)

        # ダミー画像の取得
        self.dummy_len = 0
        if dummy_path is not None:
            self.dummy_image = self.read_image(dummy_path)
            self.dummy_len = len(self.dummy_image)

        # 学習モデル出力ディレクトリの作成
        self.out_model_path = out_model_path
        if os.path.isdir(out_model_path):
            shutil.rmtree(out_model_path)

        self.d_real_labels = tf.constant(1, shape=[d_batch_size])
        self.d_fake_labels = tf.constant(0, shape=[g_batch_size])
        self.g_labels = tf.constant(1, shape=[g_batch_size])

    def read_image(self, base_path):
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

        print("image len: %d" % len(file_list))
        return image_ary

    @tf.function
    def train_gene(self):
        with tf.GradientTape() as g_tape:
            gene_image = self.gene(g_batch_size)
            g_loss = self.disc(gene_image, self.g_labels, 0.0)
        grads = g_tape.gradient(g_loss, self.gene.trainable_variables)
        self.gene_opt.apply_gradients(zip(grads, self.gene.trainable_variables))
        return g_loss, gene_image

    @tf.function
    def train_disc(self, real_img, gene_img):
        with tf.GradientTape() as tape:
            loss_real = self.disc(real_img, self.d_real_labels, 0.5)
            loss_gene = self.disc(gene_img, self.d_fake_labels, 0.5)
            loss = ((loss_real * d_batch_size) + (loss_gene * g_batch_size)) / (d_batch_size + g_batch_size)
        grads = tape.gradient(loss, self.disc.trainable_variables)
        self.disc_opt.apply_gradients(zip(grads, self.disc.trainable_variables))
        return loss

    def __call__(self):
        # 学習プロセス開始
        for itr in range(epoch_num):
            if itr % 100 == 0:
                try:
                    dirname = self.output_path + "/%06d/" % itr
                    if not os.path.isdir(dirname):
                        os.makedirs(dirname)
                    for i in range(9):
                        gene_image = self.gene()
                        img_obj = tf.reshape(gene_image, [image_x, image_y, 3])
                        img_obj = img_obj.numpy() * 255.0
                        file_name = dirname + "%02d.png" % i
                        cv2.imwrite(file_name, img_obj)
                except:
                    print("error")
                    pass
            
            # Generator
            g_loss_curr, gene_image = self.train_gene()

            # Discriminator
            rand_data = []
            rand_idx = np.random.randint(0, self.target_len, d_batch_size)
            for i in rand_idx:
                rand_data.append(self.target_image[i])
            train_np_data = np.asarray(rand_data)
            d_loss_curr = self.train_disc(train_np_data, gene_image)

            if itr % 10 == 0:
                print('Iter: {}'.format(itr))
                print(" G_loss: " + str(g_loss_curr))
                print(" D loss: " + str(d_loss_curr))

            # 学習モデル出力
            #save_path = saver.save(sess, model_path)
            #print("model save!!")

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', nargs='?', help='input image directory', required=True)
    arg_parser.add_argument('-o', nargs='?', help='output image directory', required=True)
    arg_parser.add_argument('-m', nargs='?', help='output model directory', required=True)
    args = arg_parser.parse_args()

    trainer = Trainer(args.i, args.o, args.m)
    trainer()


if __name__ == "__main__":
    main()