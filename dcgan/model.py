# coding: utf-8

import tensorflow as tf
from tensorflow.python.keras import layers as L
from tensorflow.python.keras import activations as A


class Generator(object):
	'''ランダムなベクトルから画像を生成する画像作成機
	'''
	def __init__(self, z_dim):
		super().__init__()
		self.l1 = L.Dense(6 * 6 * 512, input_shape=(z_dim,))
		self.dc1 = L.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same")
		self.dc2 = L.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")
		self.dc3 = L.Conv2DTranspose(64,  kernel_size=4, strides=2, padding="same")
		self.dc4 = L.Conv2DTranspose(3, kernel_size=4, strides=2, padding="same")

		# Convolution, Deconvolutionともに値は画像の大きさに合わせて変化させる必要がある．

		self.bn1 = L.BatchNormalization(input_shape=(512,))
		self.bn2 = L.BatchNormalization(input_shape=(256,))
		self.bn3 = L.BatchNormalization(input_shape=(128,))
		self.bn4 = L.BatchNormalization(input_shape=(64,))

		self.z_dim = z_dim

	def __call__(self, z, test=False):
		h = self.l1(z)
		# 512チャンネルをもつ、3*3のベクトルに変換する functional API
		h = L.Reshape(z.get_shape()[0], 512, 6, 6)(h)
		#print ("1",np.shape(h))
		h = A.relu(self.bn1(h))
		#print ("2",np.shape(h))
		h = A.relu(self.bn2(self.dc1(h)))
		#print ("3",np.shape(h))
		h = A.relu(self.bn3(self.dc2(h)))
		#print ("4",np.shape(h))
		h = A.relu(self.bn4(self.dc3(h)))
		#print ("5",np.shape(h))
		x = self.dc4(h)
		#print ("6",np.shape(x))
		return x

class Discriminator(object):
	'''入力された画像が偽物かどうかを判定する判別器
	'''
	def __init__(self, ):
		super().__init__()
		self.c1 = L.Conv2D(64, kernel_size=4, strides=2, padding="same", input_shape=(3,))
		self.c2 = L.Conv2D(128, kernel_size=4, strides=2, padding="same")
		self.c3 = L.Conv2D(256, kernel_size=4, strides=2, padding="same")
		self.c4 = L.Conv2D(512, kernel_size=4, strides=2, padding="same")

		self.l1 = L.Dense(2)

		self.bn1 = L.BatchNormalization(input_shape=(128,))
		self.bn2 = L.BatchNormalization(input_shape=(256,))
		self.bn3 = L.BatchNormalization(input_shape=(512,))

	def __call__(self, x, test=False):
		'''判別関数．
		return 二次元のVariable
		'''
		#print ("0",np.shape(x))
		h = A.relu(self.c1(x))
		#print ("1",np.shape(h))
		h = A.relu(self.bn1(self.c2(h)))
		#print ("2",np.shape(h))
		h = A.relu(self.bn2(self.c3(h)))
		#print ("3",np.shape(h))
		h = A.relu(self.bn3(self.c4(h)))
		#print ("4",np.shape(h))
		y = self.l1(h)
		#print ("5",np.shape(y))
		return y


