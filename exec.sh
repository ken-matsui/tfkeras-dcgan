set -eu
: $1 $2

# $1 : local OR mlengine
# $2 : 1 OR 2 OR 3 OR 4
# $3 : raw data path

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

case $2 in
"1" ) # Prepare (ML Enigne Only)
	case $1 in
	"mlengine" )
		# Create backets(Only once)
		gsutil mb -l us-central1 $ML_BACKET
		gsutil mb -l us-central1 $STAGING_BACKET
		;;
	esac
	;& # fall-through ;& is bash 4.0 over only...
"2" ) # Create Dataset
	: $3
	LOCAL_INPUT=$3
	rm -f ${LOCAL_INPUT}/.DS_Store
	trap "rm to_TFRecord.py" 0
	# Create dataset
	curl -O https://raw.githubusercontent.com/matken11235/to_TFRecord/master/to_TFRecord.py
	python to_TFRecord.py $LOCAL_INPUT
	case $1 in
	# Copy dataset to cloud storage.
	"mlengine" ) gsutil -o GSUtil:parallel_composite_upload_threshold=150M cp ./dataset.tfrecord $INPUT_FILE ;;
	esac
	;& # fall-through
"3" ) # Start Training
	case $1 in
	"local" ) python trainer.py --dataset_path ./dataset.tfrecord ;;
	"mlengine" )
		# Make unique job ID
		UNIQUE_ID=`date +%s`
		UNIQUE_NAME=`date +"%Y:%m:%d-%H:%M:%S"`
		JOB_ID="${PROJECT_NAME}_${UNIQUE_ID}"
		# Upload to ml-engine
		gcloud ml-engine jobs submit training ${JOB_ID} \
			--package-path=../DCGAN \
			--module-name=DCGAN.trainer \
			--staging-bucket=$STAGING_BACKET \
			--region=us-central1 \
			--scale-tier=BASIC_GPU \
			--runtime-version 1.4 \
			-- \
			--dataset_path=$INPUT_FILE \
			--output_path="${OUTPUT_PATH}/${UNIQUE_NAME}"
		# Print job details
		gcloud ml-engine jobs describe ${JOB_ID}
		# Open log-page
		open "https://console.cloud.google.com/ml/jobs/${JOB_ID}?project=${PROJECT_ID}"
		;;
	esac
	;; # end switch.
"4" ) # Open Tensorboard
	(sleep 10; open "http://localhost:6006") &
	case $1 in
	"local" ) tensorboard --logdir=./output/model --host=localhost ;;
	"mlengine" ) tensorboard --logdir=$OUTPUT_PATH --host=localhost ;;
	esac
	;; # end switch.
esac