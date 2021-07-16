#!/bin/bash
# Script to download and preprocess the MSCOCO data set for detection.
#
# The outputs of this script are TFRecord files containing serialized
# tf.Example protocol buffers. See create_coco_tf_record.py for details of how
# the tf.Example protocol buffers are constructed and see
# http://cocodataset.org/#overview for an overview of the dataset.
#
# usage:
#  bash object_detection/dataset_tools/preprocess_mscoco.sh \
#    /tmp/mscoco
set -e

if [ -z "$1" ]; then
  echo "usage preprocess_mscoco.sh [data dir]"
  exit
fi

# Create the output directories.
OUTPUT_DIR="${1%/}"
SCRATCH_DIR="${OUTPUT_DIR}/raw-data"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${SCRATCH_DIR}"
CURRENT_DIR=$(pwd)

cd ${SCRATCH_DIR}

TRAIN_IMAGE_DIR="${SCRATCH_DIR}/train2017"
VAL_IMAGE_DIR="${SCRATCH_DIR}/val2017"
TEST_IMAGE_DIR="${SCRATCH_DIR}/test2017"

TRAIN_ANNOTATIONS_FILE="${SCRATCH_DIR}/annotations/instances_train2017.json"
VAL_ANNOTATIONS_FILE="${SCRATCH_DIR}/annotations/instances_val2017.json"
TESTDEV_ANNOTATIONS_FILE="${SCRATCH_DIR}/annotations/image_info_test-dev2017.json"

# Build TFRecords of the image data.
cd "${CURRENT_DIR}"
python object_detection/dataset_tools/create_coco_tf_record.py \
  --logtostderr \
  --include_masks \
  --train_image_dir="${TRAIN_IMAGE_DIR}" \
  --val_image_dir="${VAL_IMAGE_DIR}" \
  --test_image_dir="${TEST_IMAGE_DIR}" \
  --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
  --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
  --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
  --output_dir="${OUTPUT_DIR}"

