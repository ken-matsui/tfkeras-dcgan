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


def parse_data(raw):
	feature = {"image": tf.FixedLenFeature((), tf.string, default_value="")}
	parsed_feature = tf.parse_single_example(raw, feature)
	image = tf.decode_raw(parsed_feature['image'], tf.uint8)
	image = tf.cast(image, tf.float32) / 255.0
	return image

def load_data(file_path):
	dataset = tf.data.TFRecordDataset(file_path)
	dataset = dataset.map(parse_data)
	dataset = dataset.shuffle(100)
	# dataset = dataset.repeat(1000) # epoch_num これしないと無限ループ
	dataset = dataset.batch(10) # batch_size
	iterator = dataset.make_initializable_iterator()
	return iterator

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
	trainer.fit(X, batch_size=10, epochs=1000)
	print("Training done.")


if __name__ == '__main__':
	tf.app.run()