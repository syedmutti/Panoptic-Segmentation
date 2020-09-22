set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# Set up the working directories.

python deeplab/eval_panoptic.py \
    --logtostderr \
    --eval_split="val" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride="8,4" \
    --eval_crop_size="1024,2048" \
    --dataset="cityscapes" \
    --colormap_type="cityscapes" \
    --checkpoint_dir="/mrtstorage/users/rehman/experiments/c2e_encoding/loss_v1/x65_b1_c200_o0.01/" \
    --eval_logdir="/mrtstorage/users/rehman/experiments/tmp/eval/Aug10/x65_b8_c1_o1_16NMS_t102"  \
    --dataset_dir="/mrtstorage/users/rehman/datasets/cityscapes/tfrecord_v3_c2e" \


