# coding: utf-8

# Inside modules
import os
# Outside modules
import cv2
import numpy as np
from tqdm import tqdm
# Other modules
from dcgan.model import Generator, Discriminator


def load_data(file_dir):
	print("Load image...")

	X = []
	for f in tqdm(os.listdir(file_dir)):
		img = cv2.imread(file_dir + file).resize(img, (96, 96)).astype(np.float32)
		X.append(img / 255.0)

	print("Image loading done.")
	print(len(X), "images read.")

	return X

def mkdir(out):
	try:
		os.mkdir(out)
		os.mkdir(out + "/model")
		os.mkdir(out + "/images")
	except:
		pass

def main():
	mkdir("out")

	gen = Generator(100)
	dis = Discriminator()

	X = load_data("learning_images/")

	print("Start training...")
	trainer = Trainer(gen, dis)
	trainer.fit(X, batch_size=100, epochs=10000)
	print("Training done.")


if __name__ == '__main__':
	main()