set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# Set up the working directories.
python deeplab/vis_panoptic_tf.py \
    --logtostderr \
    --vis_split="train" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride="8,4" \
    --vis_crop_size="1024,2048" \
    --dataset="cityscapes" \
    --colormap_type="cityscapes" \
    --initialize_last_layer=True \
    --last_layers_contain_logits_only=False \
    --checkpoint_dir="/mrtstorage/users/rehman/experiments/old_models/c500_o10_b8/" \
    --vis_logdir="/mrtstorage/users/rehman/experiments/tmp/vis/c2e/Aug10/x65_b1_33x33_August10further_c0.5_b6_72NMS_t100_raw_heat"  \
    --dataset_dir="/mrtstorage/users/rehman/datasets/cityscapes/tfrecord_v3_c2e/" \





