# coding: utf-8

# Inside modules
import os
# Outside modules
import pylab
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.keras import backend as K
# Other modules
from model import Generator, Discriminator, Hooks

flags = tf.app.flags
flags.DEFINE_string("file_path", "./dataset.tfrecord", "GCS or local paths to training data")
flags.DEFINE_string("output_path", "./out", "Output data dir")
flags.DEFINE_integer("batch_size", 1000, "batch size")
flags.DEFINE_integer("epoch_num", 10000, "epoch num")
FLAGS = flags.FLAGS


def load_data(file_path):
	def parse_data(raw):
		feature = {"image": tf.FixedLenFeature((), tf.string, default_value="")}
		parsed_feature = tf.parse_single_example(raw, feature)
		image = tf.decode_raw(parsed_feature['image'], tf.uint8)
		image = tf.reshape(image, [96, 96, 3])
		image = tf.cast(image, tf.float32) / 255.0
		return image
	# 前処理はCPUにやらせる
	with tf.device('/cpu:0'):
		dataset = tf.data.TFRecordDataset(file_path)
		dataset = dataset.map(parse_data)
		dataset = dataset.shuffle(buffer_size=1000)
		dataset = dataset.batch(FLAGS.batch_size)
		dataset = dataset.repeat(FLAGS.epoch_num)
		iterator = dataset.make_one_shot_iterator()
		next_data = iterator.get_next()
	return next_data

def fit(gen, dis, X):
	z = tf.placeholder(tf.float32, shape=[None, gen.z_dim], name="z_noise")
	x_pred = gen(z)
	y_pred1 = dis(x_pred)
	with tf.name_scope("gen_loss"):
		gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.zeros([FLAGS.batch_size], dtype=tf.int32), logits=y_pred1))
	with tf.name_scope("dis_loss1"):
		dis_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.ones([FLAGS.batch_size], dtype=tf.int32), logits=y_pred1))
	y_pred2 = dis(X)
	with tf.name_scope("dis_loss2"):
		dis_loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.zeros([FLAGS.batch_size], dtype=tf.int32), logits=y_pred2))
	with tf.name_scope("gen_train_step"):
		gen_train_step = tf.train.AdamOptimizer(0.001).minimize(gen_loss, name="gen_Adam")
	with tf.name_scope("dis_train_step"):
		dis_train_step = tf.train.AdamOptimizer(0.001).minimize(dis_loss, name="dis_Adam")
	with tf.variable_scope("global_step"):
		# globalに存在する方のglobal_stepを取得．そのためにvariable_scope．
		global_step = tf.train.get_or_create_global_step()
		# global_step = tf.Variable(-1, trainable=False, name='global_step')
		global_step_op = global_step.assign(global_step + 1)

	hooks = Hooks(x_pred, z, gen_loss, dis_loss, global_step_op, FLAGS.output_path)

	# Loggingを開始
	tf.logging.set_verbosity(tf.logging.INFO)
	train_steps = [gen_train_step, dis_train_step]
	with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
		while not sess.should_stop():
			feed_z = np.random.uniform(-1, 1, (FLAGS.batch_size, gen.z_dim))
			sess.run(train_steps, feed_dict={z: feed_z, K.learning_phase(): True})

def main(argv):
	print("Load image from", FLAGS.file_path)
	X = load_data(FLAGS.file_path)
	print(len(list(tf.python_io.tf_record_iterator(FLAGS.file_path))), "images loaded.\n")

	# Makedirs GCS or Local
	gfile.MakeDirs(FLAGS.output_path+"/model")
	gfile.MakeDirs(FLAGS.output_path+"/images")

	gen = Generator(100)
	dis = Discriminator()

	print("Start training...")
	fit(gen, dis, X)
	print("Training done.")


if __name__ == '__main__':
	tf.app.run()