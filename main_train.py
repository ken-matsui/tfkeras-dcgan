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

tf.app.flags.DEFINE_string("file_path", "./dataset.tfrecord", "GCS or local paths to training data")
tf.app.flags.DEFINE_string("output_path", "./out", "Output data dir")
tf.app.flags.DEFINE_integer("batch_size", 100, "batch size")
tf.app.flags.DEFINE_integer("epoch_num", 100, "epoch num")
FLAGS = tf.app.flags.FLAGS


def parse_data(raw):
	feature = {"image": tf.FixedLenFeature((), tf.string, default_value="")}
	parsed_feature = tf.parse_single_example(raw, feature)
	image = tf.decode_raw(parsed_feature['image'], tf.uint8)
	image = tf.reshape(image, [96, 96, 3])
	image = tf.cast(image, tf.float32) / 255.0
	return image

def load_data(file_path):
	# 前処理はCPUにやらせる
	with tf.device('/cpu:0'):
		dataset = tf.data.TFRecordDataset(file_path)
		dataset = dataset.map(parse_data)
		dataset = dataset.shuffle(buffer_size=100)
		dataset = dataset.batch(FLAGS.batch_size)
		# dataset = dataset.repeat(FLAGS.epoch_num) # これしないと無限ループ
		# iterator = dataset.make_one_shot_iterator()
		iterator = dataset.make_initializable_iterator()
		# next_data = iterator.get_next()
	# return next_data
	return iterator

def make_out_dirs(out):
	model_dir = out + "/model"
	image_dir = out + "/images"
	# GCS or Local
	gfile.MakeDirs(model_dir)
	gfile.MakeDirs(image_dir)
	return model_dir, image_dir

def main(argv):
	print("Load image from", FLAGS.file_path)
	X = load_data(FLAGS.file_path)
	print(len(list(tf.python_io.tf_record_iterator(FLAGS.file_path))), "images loaded.\n")

	gen = Generator(100)
	dis = Discriminator()
	# graph, fetches = CalcGraph(gen, dis, batch_size=100)

	out_model_dir, out_image_dir = make_out_dirs(FLAGS.output_path)

	print("Start training...")
	trainer = Trainer(gen, dis)
	trainer.fit(X, batch_size=FLAGS.batch_size, epochs=FLAGS.epoch_num)
	print("Training done.")


if __name__ == '__main__':
	tf.app.run()