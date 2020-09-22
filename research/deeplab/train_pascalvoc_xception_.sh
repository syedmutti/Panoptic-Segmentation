set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"


# Set up the working directories.

# Train 10 iterations.
NUM_ITERATIONS=100
CUDA_VISIBLE_DEVICES="1" \
python deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=15000 \
    --train_split="trainval" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size="513,513" \
    --train_batch_size=1 \
    --dataset="pascal_voc_seg" \
    --initialize_last_layer=True \
    --last_layers_contain_logits_only=False \
    --fine_tune_batch_norm=False \
    --tf_initial_checkpoint=/mrtstorage/users/rehman/datasets/pascal_voc_seg/exp/deeplabv3_pascal_train_aug/model.ckpt \
    --train_logdir=/mrtstorage/users/rehman/datasets/pascal_voc_seg/exp/train_/train_panoptic/xception_65_test/ \
    --dataset_dir=/mrtstorage/users/rehman/datasets/pascal_voc_seg/VOCdevkit/tfrecord/

