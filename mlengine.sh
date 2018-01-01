set -eu
: $1

# Define project name
PROJECT_NAME="dcgan"
# Get project ID
PROJECT_ID=`gcloud config list project --format "value(core.project)"`
# Backets URI
ML_BACKET="gs://${PROJECT_ID}-ml"
STAGING_BACKET="gs://${PROJECT_ID}-ml-staging"
# Dataset file directory for learning
INPUT_FILE="${ML_BACKET}/${PROJECT_NAME}/dataset/dataset.tfrecord"
OUTPUT_PATH="${ML_BACKET}/${PROJECT_NAME}/output"

case $1 in
"1" )
	# Install python packages(Only once)
	pip install -r ./requirements.txt
	# Create backets(Only once)
	gsutil mb -l us-central1 $ML_BACKET
	gsutil mb -l us-central1 $STAGING_BACKET
	;& # fall-through ;& bash 4.0 over only
"2" )
	: $2
	LOCAL_INPUT=$2
	trap "rm to_TFRecord.py" 0
	# Create dataset
	rm -f ${LOCAL_INPUT}/.DS_Store
	curl -O https://raw.githubusercontent.com/matken11235/to_TFRecord/master/to_TFRecord.py
	python to_TFRecord.py $LOCAL_INPUT
	# Copy learning data to cloud storage.
	gsutil cp ./dataset.tfrecord $INPUT_FILE
	;& # fall-through
"3" )
	# Make unique job ID
	UNIQUE_ID=`date +%s`
	UNIQUE_NAME=`date +"%Y:%m:%d-%H:%M:%S"`
	JOB_ID="${PROJECT_NAME}_${UNIQUE_ID}"
	# Upload to ml-engine
	gcloud ml-engine jobs submit training ${JOB_ID} \
		--package-path=../DCGAN \
		--module-name=DCGAN.main_train \
		--staging-bucket=$STAGING_BACKET \
		--region=us-central1 \
		--scale-tier=BASIC_GPU \
		--runtime-version 1.4 \
		-- \
		--file_path=$INPUT_FILE \
		--output_path="${OUTPUT_PATH}/${UNIQUE_NAME}"
	# Print job details
	gcloud ml-engine jobs describe ${JOB_ID}
	# Open log-page
	open "https://console.cloud.google.com/ml/jobs/${JOB_ID}?project=${PROJECT_ID}"
	;; # end switch.
"4" )
	python -m tensorflow.tensorboard --logdir=$OUTPUT_PATH
	;;
esac