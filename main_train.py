# coding: utf-8

# Inside modules
import os
# Outside modules
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
# Other modules
from model import Generator, Discriminator
from trainer import Trainer

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("indir_path", "./learning_images", "Input image dir")
tf.app.flags.DEFINE_string("outdir_path", "./out", "Output data dir")

def load_data(file_dir):
	X = []
	files = os.listdir(file_dir)
	files.remove(".DS_Store")
	for f in tqdm(files):
		img = cv2.imread(file_dir + f)
		img = cv2.resize(img, (96, 96))
		img = img.astype(np.float32)
		# 正規化
		X.append(img / 255.0)
	return X

def make_out_dirs(out):
	model_dir = out + "/model"
	image_dir = out + "/images"
	if out == "./out": # local(not ML Engine)
		os.makedirs(model_dir, exist_ok=True)
		os.makedirs(image_dir, exist_ok=True)
	return model_dir, image_dir

def main(argv):
	print("Load image from", FLAGS.indir_path)
	X = load_data(FLAGS.indir_path)
	print(len(X), "images loaded.")

	print()

	gen = Generator(100)
	dis = Discriminator()
	# graph, fetches = CalcGraph(gen, dis, batch_size=100)

	out_model_dir, out_image_dir = make_out_dirs(FLAGS.outdir_path)

	print("Start training...")
	trainer = Trainer(gen, dis)
	trainer.fit(X, batch_size=10, epochs=10000)
	print("Training done.")


if __name__ == '__main__':
	tf.app.run()