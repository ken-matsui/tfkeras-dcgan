```
Traceback (most recent call last):
  File "main_train.py", line 51, in <module>
    main()
  File "main_train.py", line 46, in main
    trainer.fit(X, batch_size=100, epochs=10000)
  File "/Users/matken/Documents/program/python/tfkeras/DCGAN/dcgan/trainer.py", line 42, in fit
    gen_train_step = tf.train.AdamOptimizer(0.05).minimize(gen_loss)
  File "/Users/matken/.pyenv/versions/tensorflow/lib/python3.6/site-packages/tensorflow/python/training/optimizer.py", line 350, in minimize
    ([str(v) for _, v in grads_and_vars], loss))
ValueError: No gradients provided for any variable, check your graph for ops that do not support gradients, between variables ["<tf.Variable 'dense/kernel:0' shape=(100, 18432) dtype=float32_ref>", "<tf.Variable 'dense/bias:0' shape=(18432,) dtype=float32_ref>", "<tf.Variable 'batch_normalization/gamma:0' shape=(6,) dtype=float32_ref>", "<tf.Variable 'batch_normalization/beta:0' shape=(6,) dtype=float32_ref>", "<tf.Variable 'conv2d_transpose/kernel:0' shape=(4, 4, 256, 6) dtype=float32_ref>", "<tf.Variable 'conv2d_transpose/bias:0' shape=(256,) dtype=float32_ref>", "<tf.Variable 'batch_normalization_1/gamma:0' shape=(256,) dtype=float32_ref>", "<tf.Variable 'batch_normalization_1/beta:0' shape=(256,) dtype=float32_ref>", "<tf.Variable 'conv2d_transpose_1/kernel:0' shape=(4, 4, 128, 256) dtype=float32_ref>", "<tf.Variable 'conv2d_transpose_1/bias:0' shape=(128,) dtype=float32_ref>", "<tf.Variable 'batch_normalization_2/gamma:0' shape=(128,) dtype=float32_ref>", "<tf.Variable 'batch_normalization_2/beta:0' shape=(128,) dtype=float32_ref>", "<tf.Variable 'conv2d_transpose_2/kernel:0' shape=(4, 4, 64, 128) dtype=float32_ref>", "<tf.Variable 'conv2d_transpose_2/bias:0' shape=(64,) dtype=float32_ref>", "<tf.Variable 'batch_normalization_3/gamma:0' shape=(64,) dtype=float32_ref>", "<tf.Variable 'batch_normalization_3/beta:0' shape=(64,) dtype=float32_ref>", "<tf.Variable 'conv2d_transpose_3/kernel:0' shape=(4, 4, 3, 64) dtype=float32_ref>", "<tf.Variable 'conv2d_transpose_3/bias:0' shape=(3,) dtype=float32_ref>", "<tf.Variable 'conv2d/kernel:0' shape=(4, 4, 3, 64) dtype=float32_ref>", "<tf.Variable 'conv2d/bias:0' shape=(64,) dtype=float32_ref>", "<tf.Variable 'conv2d_1/kernel:0' shape=(4, 4, 64, 128) dtype=float32_ref>", "<tf.Variable 'conv2d_1/bias:0' shape=(128,) dtype=float32_ref>", "<tf.Variable 'batch_normalization_4/gamma:0' shape=(128,) dtype=float32_ref>", "<tf.Variable 'batch_normalization_4/beta:0' shape=(128,) dtype=float32_ref>", "<tf.Variable 'conv2d_2/kernel:0' shape=(4, 4, 128, 256) dtype=float32_ref>", "<tf.Variable 'conv2d_2/bias:0' shape=(256,) dtype=float32_ref>", "<tf.Variable 'batch_normalization_5/gamma:0' shape=(256,) dtype=float32_ref>", "<tf.Variable 'batch_normalization_5/beta:0' shape=(256,) dtype=float32_ref>", "<tf.Variable 'conv2d_3/kernel:0' shape=(4, 4, 256, 512) dtype=float32_ref>", "<tf.Variable 'conv2d_3/bias:0' shape=(512,) dtype=float32_ref>", "<tf.Variable 'batch_normalization_6/gamma:0' shape=(512,) dtype=float32_ref>", "<tf.Variable 'batch_normalization_6/beta:0' shape=(512,) dtype=float32_ref>", "<tf.Variable 'dense_1/kernel:0' shape=(512, 2) dtype=float32_ref>", "<tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32_ref>"] and loss Tensor("softmax_cross_entropy_loss/value:0", shape=(), dtype=float32).
```

> https://github.com/tensorflow/tensorflow/issues/1511
```
pannous commented on 6 Dec 2016 •  edited
This can be hard to debug, and the reasons might be quite subtle,

for example if for some reason you switched arguments in softmax_cross_entropy_with_logits:

tf.nn.softmax_cross_entropy_with_logits(y,_y) to
tf.nn.softmax_cross_entropy_with_logits(_y,y)

you get the dreaded No gradients provided for any variable
```

ここに従って，入れ替えたら治った．

https://www.tensorflow.org/api_docs/python/tf/losses/softmax_cross_entropy