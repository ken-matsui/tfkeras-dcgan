# coding: utf-8

import tensorflow as tf
from tensorflow.python.keras import backend as K
import numpy as np
# import pylab

DUMP_NUM = 100

class Trainer(object):
	def __init__(self, gen, dis):
		self.gen = gen
		self.dis = dis
		self.z_dim = gen.z_dim

	def fit(self, X, epochs=10, batch_size=1000, plotting=True):
		"""
		generator と discriminator のトレーニングを行うメソッド
		:param X: 正しい画像. shape = (n_samples, width, height, channel)
		:param int epochs:
			画像全体を何回訓練させるかを表す.
		:param int batch_size:
			学習時のバッチ数.
		:param bool plotting:
			訓練中に生成画像をプロットするかどうか.
			true のときプロットを行う.
		:return:
		"""
		# Generate Noise
		# z = tf.random_uniform([batch_size, self.z_dim], minval=-1, maxval=1)
		z = tf.placeholder(tf.float32, shape=(batch_size, self.z_dim))
		# define TF graph
		x = self.gen(z)
		y_pred1 = self.dis(x)

		gen_loss = tf.losses.softmax_cross_entropy(tf.zeros(shape=(batch_size, 2), dtype=tf.float32), y_pred1)
		dis_loss = tf.losses.softmax_cross_entropy(tf.ones(shape=(batch_size, 2), dtype=tf.float32), y_pred1)

		x_data = tf.placeholder(tf.float32, shape=(batch_size, 96, 96, 3))
		y_pred2 = self.dis(x_data)

		dis_loss += tf.losses.softmax_cross_entropy(tf.zeros(shape=(batch_size, 2), dtype=tf.float32), y_pred2)

		gen_train_step = tf.train.AdamOptimizer(0.05).minimize(gen_loss)
		dis_train_step = tf.train.AdamOptimizer(0.05).minimize(dis_loss)

		init = tf.global_variables_initializer()

		n_train = len(X)
		with tf.Session() as sess:
			sess.run(init)
			# The `Iterator.string_handle()` method returns a tensor that can be evaluated
			# and used to feed the `handle` placeholder.
			for epoch in range(epochs):
				perm = np.random.permutation(n_train)
				gen_loss_sum = np.float32(0)
				dis_loss_sum = np.float32(0)
				for i in range(int(n_train / batch_size)): # for (i = X.itr.first; i != X.itr.last; ++i)
					# Load true data form dataset
					idx = perm[i * batch_size:(i + 1) * batch_size]
					x_datas = []
					for j in idx:
						x_datas.append(X[j])
					x_batch = np.array(x_datas, dtype=np.float32)
					z_data = np.random.uniform(-1, 1, (batch_size, self.z_dim))
					gen_fd = { z: z_data, K.learning_phase(): 1 }
					dis_fd = { z: z_data, x_data: x_batch, K.learning_phase(): 1 }
					# TODO: tf.random_uniformをfeed_dictを渡す．
					_, gen_loss_val = sess.run([gen_train_step, gen_loss], feed_dict=gen_fd)
					_, dis_loss_val = sess.run([dis_train_step, dis_loss], feed_dict=dis_fd)
					gen_loss_sum += gen_loss_val
					dis_loss_sum += dis_loss_val
				print("\tepoch, gen_loss, dis_loss = %6d: %6.3f, %6.3f" % (epoch+1, gen_loss_sum, dis_loss_sum))

				# if plotting and epoch%DUMP_NUM == 0:
				# 	saver = tf.train.Saver()
				# 	saver.save(sess, 'out/model')

					# pylab.rcParams['figure.figsize'] = (16.0,16.0)
					# pylab.clf()
					# n_row = 5
					# s = n_row**2
					# z = tf.random_uniform([batch_size, self.z_dim], minval=-1, maxval=1)
					# x = self.gen(z, test=True)
					# y = self.dis(x)
