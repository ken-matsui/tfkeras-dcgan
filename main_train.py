# coding: utf-8

# Inside modules
import os
# Outside modules
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
# Other modules
from model import Generator, Discriminator
from trainer import Trainer

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("file_path", "./dataset.tfrecord", "GCS or local paths to training data")
tf.app.flags.DEFINE_string("output_path", "./out", "Output data dir")

def _load_data(filename):
	image_string = tf.read_file(filename)
	# Once you upgrade to TF 1.2 decode_png will decode both jpg as well as png and vice versa.
	# decode_image cannot be used along with resize since it also supports .gif .
	img = tf.image.decode_png(image_string, channels=3)
	# img = cv2.resize(img, (96, 96))
	img = tf.image.resize_images(img, [96, 96])
	return img / 255.0

def load_data(file_path):
	# print("Load image from", filedir)
	# filenames = gfile.ListDirectory(filedir)
	# filenames.remove(".DS_Store")
	# dataset = tf.data.Dataset.from_tensor_slices(filenames) \
	# 			.map(_load_data) \
	# 			.shuffle(len(filenames)) \
	# 			.batch(10)
	# print(len(filenames), "images loaded.\n")
	# iterator = dataset.make_one_shot_iterator()
	# return iterator.get_next()
	# X = []
	# filenames = os.listdir(filedir)
	# filenames.remove(".DS_Store")
	# for f in filenames:
	# 	img = cv2.imread(filedir + '/' + f)
	# 	img = cv2.resize(img, (96, 96))
	# 	img = img.astype(np.float32)
	# 	# 正規化
	# 	X.append(img / 255.0)
	# return X
	# 読み込み対象のファイルをqueueに詰める: TFRecordReaderはqueueを利用してファイルを読み込む
	file_name_queue = tf.train.string_input_producer([file_path])
	# TFRecordsファイルを読み込む為、TFRecordReaderオブジェクトを生成
	reader = tf.TFRecordReader()
	# 読み込み: ファイルから読み込み、dataset_serializedに格納する
	_, dataset_serialized = reader.read(file_name_queue)
	# Deserialize: return: Tensor Object
	features = tf.parse_single_example(
				dataset_serialized,
				features={
					"image": tf.FixedLenFeature([], tf.string),
					"height": tf.FixedLenFeature([], tf.int64),
					"width": tf.FixedLenFeature([], tf.int64),
					"channels": tf.FixedLenFeature([], tf.int64),
				})
	with tf.Session() as sess:
		sess.run(tf.local_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		try:
			# evalにより，Tensor Objectから実数値へ変換
			height = tf.cast(features["height"], tf.int32).eval()
			width = tf.cast(features["width"], tf.int32).eval()
			channels = tf.cast(features["channels"], tf.int32).eval()
			# 学習時に取り出すため，実数値に変換しない
			img = tf.reshape(tf.decode_raw(features["image"], tf.uint8), tf.stack([height, width, channels]))
		finally:
			coord.request_stop()
			coord.join(threads)
	# 正規化
	img = tf.cast(img, tf.float32) / 255.0
	return img

def make_out_dirs(out):
	model_dir = out + "/model"
	image_dir = out + "/images"
	# GCS or Local
	gfile.MakeDirs(model_dir)
	gfile.MakeDirs(image_dir)
	return model_dir, image_dir

def main(argv):
	X = load_data(FLAGS.file_path)

	gen = Generator(100)
	dis = Discriminator()
	# graph, fetches = CalcGraph(gen, dis, batch_size=100)

	out_model_dir, out_image_dir = make_out_dirs(FLAGS.output_path)

	print("Start training...")
	trainer = Trainer(gen, dis)
	trainer.fit(X, batch_size=10, epochs=10000)
	print("Training done.")


if __name__ == '__main__':
	tf.app.run()