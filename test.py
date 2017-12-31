import tensorflow as tf
sess = tf.InteractiveSession()

dataset = tf.data.Dataset.range(10).shuffle(10).batch(2)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
while True:
	try:
		print(sess.run(next_element))
	except tf.errors.OutOfRangeError:
		break

sess.close()

''': ex) Output
[6 0]
[9 5]
[1 8]
[3 7]
[2 4]
'''