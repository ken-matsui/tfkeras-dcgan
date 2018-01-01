set -eu
: $1

case $1 in
"1" )
	# Install python packages(Only once)
	pip install -r ./requirements.txt
	;&
"2" )
	# Create dataset
	rm -f ./mnist-images/.DS_Store
	curl -O https://raw.githubusercontent.com/matken11235/to_TFRecord/master/to_TFRecord.py
	python to_TFRecord.py ./mnist-images
	rm to_TFRecord.py
	;&
"3" )
	python main_train.py --file_path ./dataset.tfrecord
	;;
esac