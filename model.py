# coding: utf-8

import tensorflow as tf
from tensorflow.python.keras import layers as L
from tensorflow.python.keras import activations as A

class Generator(object):
	'''ランダムなベクトルから画像を生成する画像作成機
	'''
	def __init__(self, z_dim):
		super(Generator, self).__init__()
		with tf.variable_scope("generator"):
			self.l1 = L.Dense(512 * 6 * 6, input_shape=(100, z_dim))

			self.dc1 = L.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same")
			self.dc2 = L.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")
			self.dc3 = L.Conv2DTranspose(64,  kernel_size=4, strides=2, padding="same")
			self.dc4 = L.Conv2DTranspose(3, kernel_size=4, strides=2, padding="same")

			self.bn1 = L.BatchNormalization(input_shape=(512,))
			self.bn2 = L.BatchNormalization(input_shape=(256,))
			self.bn3 = L.BatchNormalization(input_shape=(128,))
			self.bn4 = L.BatchNormalization(input_shape=(64,))

			self.z_dim = z_dim

	def __call__(self, z, test=False):
		# print(z.name, z.shape)
		h = self.l1(z)
		# print(h.name, h.shape)
		h = tf.reshape(h, [z.get_shape()[0], 6, 6, 512])
		# print(h.name, h.shape)
		h = A.relu(self.bn1(h))
		# print(h.name, h.shape)
		h = A.relu(self.bn2(self.dc1(h)))
		# print(h.name, h.shape)
		h = A.relu(self.bn3(self.dc2(h)))
		# print(h.name, h.shape)
		h = A.relu(self.bn4(self.dc3(h)))
		# print(h.name, h.shape)
		x = self.dc4(h)
		# print(x.name, x.shape)
		return x

class Discriminator(object):
	'''入力された画像が偽物かどうかを判定する判別器
	'''
	def __init__(self):
		super(Discriminator, self).__init__()
		with tf.variable_scope("discriminator"):
			self.c1 = L.Conv2D(64, kernel_size=4, strides=2, padding="same", input_shape=(None, 96, 3))
			self.c2 = L.Conv2D(128, kernel_size=4, strides=2, padding="same")
			self.c3 = L.Conv2D(256, kernel_size=4, strides=2, padding="same")
			self.c4 = L.Conv2D(512, kernel_size=4, strides=2, padding="same")

			self.bn1 = L.BatchNormalization(input_shape=(128,))
			self.bn2 = L.BatchNormalization(input_shape=(256,))
			self.bn3 = L.BatchNormalization(input_shape=(512,))

			self.l1 = L.Dense(2)

		# ReLU test
		# with tf.Session() as sess:
		# 	print(sess.run(A.relu([[-1, 1, 2], [-1, -3, 4]])))

	def __call__(self, x, test=False):
		'''判別関数．
		:return: 二次元のVariable
		'''
		# print(x.name, x.shape)
		h = A.relu(self.c1(x))
		# print(h.name, h.shape)
		h = A.relu(self.bn1(self.c2(h)))
		# print(h.name, h.shape)
		h = A.relu(self.bn2(self.c3(h)))
		# print(h.name, h.shape)
		h = A.relu(self.bn3(self.c4(h)))
		# print(h.name, h.shape)
		h = tf.reshape(h, [x.get_shape()[0], 6 * 6 * 512])
		# print(h.name, h.shape)
		y = self.l1(h)
		# print(y.name, y.shape)
		return y

class CalcGraph(object):
	'''計算グラフ
	:return: graph, gen_fetches(step, loss), dis_fetches(step, loss)
	'''
	def __new__(cls, gen, dis, batch_size):
		# Generate Noise
		z = tf.random_uniform([batch_size, gen.z_dim], minval=-1, maxval=1)
		# define TF graph
		x = gen(z)
		y_pred1 = dis(x)

		gen_loss = tf.losses.softmax_cross_entropy(tf.zeros(shape=(batch_size, 512, 6, 2), dtype=tf.float32), y_pred1)
		dis_loss = tf.losses.softmax_cross_entropy(tf.ones(shape=(batch_size, 512, 6, 2), dtype=tf.float32), y_pred1)

		x_data = tf.placeholder(tf.float32, shape=(batch_size, 96, 96, 3))
		y_pred2 = dis(x_data)

		dis_loss += tf.losses.softmax_cross_entropy(tf.zeros(shape=(batch_size, 6, 6, 2), dtype=tf.float32), y_pred2)

		gen_train_step = tf.train.AdamOptimizer(0.05).minimize(gen_loss)
		dis_train_step = tf.train.AdamOptimizer(0.05).minimize(dis_loss)

		init = tf.global_variables_initializer()

		return init, [gen_train_step, gen_loss], [dis_train_step, dis_loss]

