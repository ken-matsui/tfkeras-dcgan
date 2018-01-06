# Inside modules
import sys
import time
# Outside modules
import numpy as np
import tensorflow as tf
from tensorflow.python.training.basic_session_run_hooks import _as_graph_element


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
