import tensorflow as tf
from PIL import Image
import numpy as np
# sess = tf.InteractiveSession()

# 読み込み対象のファイルをqueueに詰める: TFRecordReaderはqueueを利用してファイルを読み込む
file_name_queue = tf.train.string_input_producer(["./dataset.tfrecords"])
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

with tf.Session() as sess:
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	img = sess.run(img)
	Image.fromarray(np.uint8(img)).show()
