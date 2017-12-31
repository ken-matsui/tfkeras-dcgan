set -eu
: $1

# Define project name
PROJECT_NAME="dcgan"
# Get project ID
PROJECT_ID=`gcloud config list project --format "value(core.project)"`
# Backets URI
ML_BACKET="gs://${PROJECT_ID}-ml"
STAGING_BACKET="gs://${PROJECT_ID}-ml-staging"
# Image file dir for learning
INPUT_FILE="${ML_BACKET}/data/${PROJECT_NAME}/dataset.tfrecord"

case $1 in
"1" )
	# Install python packages(Only once)
	pip install -r ./requirements.txt
	# Create backets(Only once)
	gsutil mb -l us-central1 $ML_BACKET
	gsutil mb -l us-central1 $STAGING_BACKET
	;; # fall-through ;&
"2" )
	# Create dataset
	rm -f ./learning_images/.DS_Store
	curl -O https://raw.githubusercontent.com/matken11235/to_TFRecord/master/to_TFRecord.py
	python to_TFRecord.py ./learning_images
	rm to_TFRecord.py
	# Copy learning data to cloud storage.
	gsutil cp ./dataset.tfrecord $INPUT_FILE
	;; # fall-through ;& bash 4.0 over only
"3" )
	# Make unique job name
	JOB_NAME="${PROJECT_NAME}_`date +%s`"
	# Upload to ml-engine
	gcloud ml-engine jobs submit training ${JOB_NAME} \
		--package-path=../DCGAN \
		--module-name=DCGAN.main_train \
		--staging-bucket=$STAGING_BACKET \
		--region=us-central1 \
		--config=./config.yaml \
		--runtime-version 1.4 \
		-- \
		--file_path=$INPUT_FILE \
		--output_path="${ML_BACKET}/${PROJECT_NAME}/${JOB_NAME}"
	;; # end switch.
esac