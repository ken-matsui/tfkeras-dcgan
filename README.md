# tfkeras_DCGAN

## Usage
Please execute on bash.

```
$ chmod +x exec.sh
$ ./exec.sh (local or mlengine) step_num
                  OR
$ bash exec.sh (local or mlengine) step_num
```
bash 4.0以降の文法

```
$ bash -version
GNU bash, version 3.2.57(1)-release (x86_64-apple-darwin16)
Copyright (C) 2007 Free Software Foundation, Inc.
$ brew install bash
$ sudo vim /etc/shells
```

```
#/bin/bash
/usr/local/bin/bash
```

```
$ exec -l $SHELL
$ bash -verion
GNU bash, バージョン 4.4.12(1)-release (x86_64-apple-darwin16.3.0)
Copyright (C) 2016 Free Software Foundation, Inc.
ライセンス GPLv3+: GNU GPL バージョン 3 またはそれ以降 <http://gnu.org/licenses/gpl.html>

This is free software; you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.
```


## 参考
> 複雑な前処理も簡単に！TensorFlowのDataset APIの使い方
https://deepage.net/tensorflow/2017/07/18/tensorflow-dataset-api.html

> TensorFlowのデータフォーマットTFRecordの書き込みと読み込み方法
https://deepage.net/tensorflow/2017/10/07/tfrecord.html

> TensorFlowの計算グラフ内の変数tf.Variableの使い方
https://deepage.net/tensorflow/2017/06/02/tensorflow-variable.html

> 定数・変数・プレースホルダの利用方法を理解する。：Tensorflow入門の入門2／文系向け
http://arakan-pgm-ai.hatenablog.com/entry/2017/05/06/214113

> TensorFlow の名前空間を理解して共有変数を使いこなす
https://qiita.com/TomokIshii/items/ffe999b3e1a506c396c8

> TensorFlowのキーコンセプト: Opノード、セッション、変数
https://qiita.com/yanosen_jp/items/70e6d6afc36e1c0a3ef3

> Github [tensorflow/tensorflow/python/training/basic_session_run_hooks.py]
https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/training/basic_session_run_hooks.py

> Tensorflow API r1.4 tf.train.SessionRunArgs
https://www.tensorflow.org/api_docs/python/tf/train/SessionRunArgs

> Tensorflow Develop Performance Guide
https://www.tensorflow.org/performance/performance_guide

> Tensorflow Develop r1.3 Importing Data
https://www.tensorflow.org/versions/r1.3/programmers_guide/datasets

> Tensoflow API r0.12 class Distributed execution [tf.train.MonitoredSession]
https://www.tensorflow.org/versions/r0.12/api_docs/python/train/distributed_execution#MonitoredSession

> Tensorflow API r1.4 Training [Training Hooks]
https://www.tensorflow.org/api_guides/python/train#Training_Hooks

> Tensorflow API r1.4 tf.train.MonitoredTrainingSession
https://www.tensorflow.org/api_docs/python/tf/train/MonitoredTrainingSession

> Tensorflow Develop TensorBoard: Visualizing Learning
https://www.tensorflow.org/get_started/summaries_and_tensorboard

> Tensorflow Develop TensorBoard Histogram Dashboard
https://www.tensorflow.org/get_started/tensorboard_histograms

> あらゆるデータを可視化するTensorBoard徹底入門
https://deepage.net/tensorflow/2017/04/25/tensorboard.html

