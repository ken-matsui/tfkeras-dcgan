# Outside modules
import tensorflow as tf
from tensorflow.python.keras import layers as L
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
