# coding: utf-8

import tensorflow as tf
from tensorflow.python.keras import backend as K
import numpy as np
# import pylab

DUMP_NUM = 100

class Hoge(tf.train.SessionRunHook):
	def begin(self):
		self._step = -1

	def before_run(self, run_context):
		self._step += 1
		self._start_time = time.time()
		return tf.train.SessionRunArgs(add)

	def after_run(self, run_context, run_values):
		duration = time.time() - self._start_time
		result = run_values.results
		if self._step % 10 == 0:
			format_str = 'RESULT: {}, STEP:{}, {:%Y-%m-%d %H:%M:%S}, {:.2}'
			print(format_str.format(result, self._step, datetime.now(), duration))

def cross_entropy(labels, logits):
	return -tf.reduce_sum(labels*tf.log(tf.clip_by_value(logits, 1e-10, 1.0)))

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
		# with tf.Graph().as_default():
		# Generate Noise
		z = tf.placeholder(tf.float32, shape=(batch_size, self.z_dim))
		# define TF graph
		x = self.gen(z)
		y_pred1 = self.dis(x)

		gen_loss = cross_entropy(tf.zeros(shape=(batch_size, 2), dtype=tf.float32), y_pred1)
		# gen_loss = tf.losses.softmax_cross_entropy(tf.zeros(shape=(batch_size, 2), dtype=tf.int32), y_pred1)
		# gen_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.zeros(shape=(batch_size, 2), dtype=tf.float32), logits=y_pred1))
		dis_loss = tf.losses.softmax_cross_entropy(tf.ones(shape=(batch_size, 2), dtype=tf.float32), y_pred1)
		# dis_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.ones(shape=(batch_size, 2), dtype=tf.float32), logits=y_pred1))

		x_data = tf.placeholder(tf.float32, shape=(batch_size, 96, 96, 3))
		y_pred2 = self.dis(x_data)

		dis_loss += tf.losses.softmax_cross_entropy(tf.zeros(shape=(batch_size, 2), dtype=tf.float32), y_pred2)
		# dis_loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.zeros(shape=(batch_size, 2), dtype=tf.float32), logits=y_pred2))

		gen_train_step = tf.train.AdamOptimizer(0.001).minimize(gen_loss)
		dis_train_step = tf.train.AdamOptimizer(0.001).minimize(dis_loss)

		init = tf.global_variables_initializer()
		next_element = X.get_next()

		with tf.Session() as sess:
			sess.run(init)
			for epoch in range(epochs):
				sess.run(X.initializer)
				gen_loss_sum = np.float32(0)
				dis_loss_sum = np.float32(0)
				while True:
					try:
						X_data = sess.run(next_element)
						z_data = np.random.uniform(-1, 1, (batch_size, self.z_dim))
						train_fd = { z: z_data, x_data: X_data, K.learning_phase(): 1 }
						_, gen_loss_val = sess.run([gen_train_step, gen_loss], feed_dict=train_fd)
						# _, dis_loss_val = sess.run([dis_train_step, dis_loss], feed_dict=train_fd)
						gen_loss_sum += gen_loss_val
						# dis_loss_sum += dis_loss_val
						print(gen_loss_val)
					except tf.errors.OutOfRangeError:
						break
				print("\tepoch, gen_loss, dis_loss = %6d: %6.3f, %6.3f" % (epoch+1, gen_loss_sum, dis_loss_sum))

		# with tf.train.MonitoredTrainingSession(
		# 	checkpoint_dir='./out/model',
		# 	hooks=[Hoge()]
		# 	) as sess:
		# 	while not sess.should_stop():
		# 		gen_loss_sum = np.float32(0) # callbackで，epoch beforeで処理する
		# 		dis_loss_sum = np.float32(0)
		# 		X_data = sess.run(next_element)
		# 		z_data = np.random.uniform(-1, 1, (batch_size, self.z_dim))
		# 		train_fd = { z: z_data, x_data: X_data, K.learning_phase(): 1 }
		# 		_, gen_loss_val = sess.run([gen_train_step, gen_loss], feed_dict=train_fd)
		# 		_, dis_loss_val = sess.run([dis_train_step, dis_loss], feed_dict=train_fd)
		# 		gen_loss_sum += gen_loss_val
		# 		dis_loss_sum += dis_loss_val

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
