# coding: utf-8

import pylab
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers as L
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import activations as A

class Generator(object):
	'''ランダムなベクトルから画像を生成する画像作成機
	'''
	def __init__(self, z_dim):
		super(Generator, self).__init__()
		self.l1 = L.Dense(6 * 6 * 512, input_shape=(100, z_dim))

		self.dc1 = L.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same")
		self.dc2 = L.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")
		self.dc3 = L.Conv2DTranspose(64,  kernel_size=4, strides=2, padding="same")
		self.dc4 = L.Conv2DTranspose(3, kernel_size=4, strides=2, padding="same")

		self.bn1 = L.BatchNormalization(input_shape=(512,))
		self.bn2 = L.BatchNormalization(input_shape=(256,))
		self.bn3 = L.BatchNormalization(input_shape=(128,))
		self.bn4 = L.BatchNormalization(input_shape=(64,))

		self.z_dim = z_dim

	def __call__(self, z):
		with tf.variable_scope("Generator"):
			# print(z.name, z.shape)
			h = self.l1(z)
			# print(h.name, h.shape)
			h = tf.reshape(h, [-1, 6, 6, 512])
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
			# print(x.name, x.shape, "\n")
		return x

class Discriminator(object):
	'''入力された画像が偽物かどうかを判定する判別器
	'''
	def __init__(self):
		super(Discriminator, self).__init__()
		self.c1 = L.Conv2D(64, kernel_size=4, strides=2, padding="same", input_shape=(None, 96, 3))
		self.c2 = L.Conv2D(128, kernel_size=4, strides=2, padding="same")
		self.c3 = L.Conv2D(256, kernel_size=4, strides=2, padding="same")
		self.c4 = L.Conv2D(512, kernel_size=4, strides=2, padding="same")

		self.bn1 = L.BatchNormalization(input_shape=(128,))
		self.bn2 = L.BatchNormalization(input_shape=(256,))
		self.bn3 = L.BatchNormalization(input_shape=(512,))

		self.l1 = L.Dense(2)

	def __call__(self, x):
		with tf.variable_scope("Discriminator"):
			# print(x.name, x.shape)
			h = A.relu(self.c1(x))
			# print(h.name, h.shape)
			h = A.relu(self.bn1(self.c2(h)))
			# print(h.name, h.shape)
			h = A.relu(self.bn2(self.c3(h)))
			# print(h.name, h.shape)
			h = A.relu(self.bn3(self.c4(h)))
			# print(h.name, h.shape)
			h = tf.reshape(h, [-1, 6 * 6 * 512])
			# print(h.name, h.shape)
			y = self.l1(h)
			# print(y.name, y.shape, "\n")
		return y

DUMP_NUM = 100
class Hooks(object):
	'''MonitoredTrainingSessionに渡すためのhooks
	:return: hooks
	'''
	def __new__(cls, x_pred, z, gen_loss, dis_loss, global_step_op, output_path):
		class ImageCSListerner(tf.train.CheckpointSaverListener):
			def after_save(self, session, global_step_value):
				pylab.rcParams['figure.figsize'] = (16.0, 16.0)
				pylab.clf()
				row = 5
				s = row**2
				feed_z = np.random.uniform(-1, 1, 100 * s).reshape(-1, 100).astype(np.float32)
				x_val = session.run(x_pred, feed_dict={z: feed_z, K.learning_phase(): False})
				xs = np.reshape(x_val, (-1, 3, 96, 96))
				for i in range(s):
					tmp = xs[i].transpose(1, 2, 0)
					tmp = np.clip(tmp, 0.0, 1.0)
					pylab.subplot(row, row, i+1)
					pylab.imshow(tmp)
					pylab.axis("off")
				filename = "%s/epoch-%s.png" % (output_path+"/images", global_step_value)
				tf.logging.info("Plotting image for %s into %s." % (global_step_value, filename))
				pylab.savefig(filename, dip=100)

		log_format = "Iter %4d: gen_loss=%6.8f, dis_loss=%6.8f"
		# Hookの定義
		hooks = [
			# 与えたlossがNaNを出せば例外を発生させるHook
			tf.train.NanTensorHook(gen_loss),
			tf.train.NanTensorHook(dis_loss),
			# 指定イテレート回数分ログを出力するHook
			tf.train.LoggingTensorHook(
				every_n_iter=1,
				tensors={
					"step": global_step_op,
					"gen_loss": gen_loss,
					"dis_loss": dis_loss
				},
				formatter=lambda t: log_format % (
					t["step"],
					t["gen_loss"],
					t["dis_loss"]
				)
			), # 関数から，このクラスのselfにぶち込めば，state保管できるんじゃない？？そこから，指定回数文の総和を持ってきて，出力すればBeautiful!
			tf.train.CheckpointSaverHook(
				checkpoint_dir=output_path+"/model",
				save_steps=DUMP_NUM,
				listeners=[ImageCSListerner()]
			),
		]
		return hooks

