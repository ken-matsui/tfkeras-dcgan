set -eu
: $1

case $1 in
"1" )
	# Install python packages(Only once)
	# pip install -r ./requirements.txt
	;&
"2" )
	# Create dataset
	rm -f ./mnist_images/.DS_Store
	trap "rm to_TFRecord.py" 0
	curl -O https://raw.githubusercontent.com/matken11235/to_TFRecord/master/to_TFRecord.py
	python to_TFRecord.py ./mnist_images
	;&
"3" )
	python trainer.py --file_path ./dataset.tfrecord
	;;
"4" )
	(sleep 10; open "http://localhost:6006") &
	tensorboard --logdir=./out/model
	;;
esac