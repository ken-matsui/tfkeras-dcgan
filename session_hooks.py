# Inside modules
import sys
import time
# Outside modules
import matplotlib
matplotlib.use('agg')
import pylab
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.keras import backend as K
from tensorflow.python.training.basic_session_run_hooks import _as_graph_element


class ImageCSListerner(tf.train.CheckpointSaverListener):
	def __init__(self, z, x_pred, output_path):
		# 参照渡しの意
		self.z = z
		self.x_pred = x_pred
		self.output_path = output_path

	def after_save(self, session, global_step_value):
		pylab.rcParams['figure.figsize'] = (16.0, 16.0)
		pylab.clf()
		row = 5
		s = row**2
		feed_z = np.random.uniform(-1, 1, 100 * s).reshape(-1, 100).astype(np.float32)
		x_val = session.run(self.x_pred, feed_dict={self.z: feed_z, K.learning_phase(): False})
		xs = np.reshape(x_val, (-1, 3, 96, 96))
		for i in range(s):
			tmp = xs[i].transpose(1, 2, 0)
			tmp = np.clip(tmp, 0.0, 1.0)
			pylab.subplot(row, row, i+1)
			pylab.imshow(tmp)
			pylab.axis("off")
		filename = "/iter%s.png"
		output_path = self.output_path + "/images" + filename
		tf.logging.info("Plotting image for %s into %s." % (global_step_value, output_path % ("")))
		tmp_path = "." + filename % ("-" + str(global_step_value))
		pylab.savefig(tmp_path, dip=100)
		# GCS or Local
		gfile.Copy(tmp_path, output_path % ("-" + str(global_step_value)), overwrite=True)
		gfile.Remove(tmp_path)

class EpochLoggingTensorHook(tf.train.SessionRunHook):
	def __init__(self, iters_per_epoch, global_step_op, gen_loss, dis_loss):
		"""
		:iters_per_epoch: epoch毎のiterator数(iters_per_epoch=10: 10Iter == 1Epoch)
		*********************************************
		* Iteratorとstepは同じ意味として扱っている．		*
		* 明確にglobal_stepを示している場合は，stepとし，	*
		* そうでない場合は全てIteratorと呼ぶこととする．	*
		* ただし，Userから見えるのは全てIteratorとする．	*
		*********************************************
		"""
		self._tensors = {"step": global_step_op,
						 "gen_loss": gen_loss,
						 "dis_loss": dis_loss}
		self._iters_per_epoch = iters_per_epoch

	def begin(self):
		self._iter_count = 0
		self._epoch_count = 1
		self._current_tensors = {tag: _as_graph_element(tensor)
								 for (tag, tensor) in self._tensors.items()}
		self._gen_loss_sum = np.float32(0)
		self._dis_loss_sum = np.float32(0)

	def before_run(self, run_context):
		return tf.train.SessionRunArgs(self._current_tensors)

	def after_run(self, run_context, run_values):
		_ = run_context
		step = run_values.results["step"]
		gen_loss = run_values.results["gen_loss"]
		dis_loss = run_values.results["dis_loss"]
		self._gen_loss_sum += gen_loss
		self._dis_loss_sum += dis_loss

		if (self._iter_count != 0) and (self._iter_count % self._iters_per_epoch == 0):
			epoch_log_format = "Epoch %4d: gen_loss=%6.8f, dis_loss=%6.8f"
			tf.logging.info(epoch_log_format % (self._epoch_count, self._gen_loss_sum, self._dis_loss_sum))
			self._epoch_count += 1
			self._gen_loss_sum = np.float32(0)
			self._dis_loss_sum = np.float32(0)
		iter_log_format = "Iter %4d: gen_loss=%6.8f, dis_loss=%6.8f\r"
		sys.stdout.write(iter_log_format % (step, gen_loss, dis_loss))
		sys.stdout.flush()
		time.sleep(0.01)
		self._iter_count += 1
