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
from model import Generator, Discriminator

flags = tf.app.flags
flags.DEFINE_string("file_path", "./dataset.tfrecord", "GCS or local paths to training data")
flags.DEFINE_string("output_path", "./out", "Output data dir")
flags.DEFINE_integer("batch_size", 1000, "batch size")
flags.DEFINE_integer("epoch_num", 100, "epoch num")
FLAGS = flags.FLAGS

DUMP_NUM = 10

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
		dataset = dataset.repeat(FLAGS.epoch_num) # これしないと無限ループ
		iterator = dataset.make_one_shot_iterator()
		# iterator = dataset.make_initializable_iterator()
		next_data = iterator.get_next()
		next_data = tf.reshape(next_data, [FLAGS.batch_size, 96, 96, 3])
	return next_data
	# return iterator

def fit(gen, dis, X):
	# Generate Noise
	z = tf.placeholder(tf.float32, shape=[None, gen.z_dim], name="z_noise")
	# define TF graph
	x_pred = gen(z)
	y_pred1 = dis(x_pred)

	gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.zeros([FLAGS.batch_size], dtype=tf.int32), logits=y_pred1, name="gen_loss"))
	dis_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.ones([FLAGS.batch_size], dtype=tf.int32), logits=y_pred1, name="dis_loss1"))

	y_pred2 = dis(X)

	dis_loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.zeros([FLAGS.batch_size], dtype=tf.int32), logits=y_pred2, name="dis_loss2"))

	gen_train_step = tf.train.AdamOptimizer(0.001).minimize(gen_loss)
	dis_train_step = tf.train.AdamOptimizer(0.001).minimize(dis_loss)

	with tf.variable_scope("global_step"):
		# global_step = tf.Variable(-1, trainable=False, name='global_step')
		global_step = tf.train.get_or_create_global_step()
		global_step_op = global_step.assign(global_step + 1)

	class ImageCSListerner(tf.train.CheckpointSaverListener):
		def after_save(self, session, global_step_value):
			pylab.rcParams['figure.figsize'] = (16.0, 16.0)
			pylab.clf()
			row = 5
			s = row**2
			x_val = session.run(x_pred,
				feed_dict={
					z: np.random.uniform(-1, 1, 100 * s).reshape(-1, 100).astype(np.float32),
					K.learning_phase(): False
				}
			)
			xs = np.reshape(x_val, (-1, 3, 96, 96))
			for i in range(s):
				tmp = xs[i].transpose(1, 2, 0)
				tmp = np.clip(tmp, 0.0, 1.0)
				pylab.subplot(row, row, i+1)
				pylab.imshow(tmp)
				pylab.axis("off")
			filename = "%s/epoch-%s.png" % (FLAGS.output_path+"/images", global_step_value)
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
			checkpoint_dir=FLAGS.output_path+"/model",
			save_steps=DUMP_NUM,
			listeners=[ImageCSListerner()]
		),
	]

	# Loggingを開始
	tf.logging.set_verbosity(tf.logging.INFO)
	with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
		while not sess.should_stop():
			sess.run([gen_train_step, dis_train_step],
				feed_dict={
					z: np.random.uniform(-1, 1, (FLAGS.batch_size, gen.z_dim)),
					K.learning_phase(): True
				}
			)

def main(argv):
	print("Load image from", FLAGS.file_path)
	X = load_data(FLAGS.file_path)
	print(len(list(tf.python_io.tf_record_iterator(FLAGS.file_path))), "images loaded.\n")

	# Makedirs GCS or Local
	gfile.MakeDirs(FLAGS.output_path+"/model")
	gfile.MakeDirs(FLAGS.output_path+"/images")

	gen = Generator(100)
	dis = Discriminator()
	# graph, fetches = CalcGraph(gen, dis, batch_size=100)

	print("Start training...")
	fit(gen, dis, X)
	print("Training done.")


if __name__ == '__main__':
	tf.app.run()