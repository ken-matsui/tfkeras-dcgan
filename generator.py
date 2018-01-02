saver = tf.train.Saver()

ckpt = tf.train.get_checkpoint_state('./')

if ckpt:
	last_model = ckpt.model_checkpoint_path
	print "load " + last_model
	saver.restore(sess, last_model)

	from PIL import Image
	import numpy as np

	new_img = Image.open('./new_data_2.png').convert('L')
	new_img = 1.0 - np.asarray(new_img, dtype="float32") / 255
	new_img = new_img.reshape((1,784))

	prediction = tf.argmax(y_conv,1)
	print("result: %g"%prediction.eval(feed_dict={x: new_img, keep_prob: 1.0}, session=sess))

else:
	# 学習
	saver.save(sess, "model.ckpt")
