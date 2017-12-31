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
INPUT_FILE="${ML_BACKET}/data/${PROJECT_NAME}"

case $1 in
"1" )
	# Create backets(Only once)
	gsutil mb -l us-central1 $ML_BACKET
	gsutil mb -l us-central1 $STAGING_BACKET
	;& # fall-through
"2" )
	# Copy learning data to cloud storage.
	gsutil -m cp learning_images/* -o $INPUT_FILE
	;& # fall-through
"3" )
	# Make unique job name
	JOB_NAME="${PROJECT_NAME}_`date +%s`"

	gcloud ml-engine jobs submit training ${JOB_NAME} \
		--package-path=../DCGAN \
		--module-name=DCGAN.main_train \
		--staging-bucket=$STAGING_BACKET \
		--region=us-central1 \
		--config=./config.yaml \
		--packages=tqdm-4.19.5-py2.py3-none-any.whl \
		--runtime-version 1.4 \
		-- \
		--file_path=$INPUT_FILE \
		--output_path="${ML_BACKET}/${PROJECT_NAME}/${JOB_NAME}"
	;; # end switch.
esac