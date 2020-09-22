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
    --model_variant="mobilenet_v3_large_seg" \
    --vis_crop_size="1024,2048" \
    --aspp_with_squeeze_and_excitation=1 \
    --image_se_uses_qsigmoid=1 \
    --image_pyramid=1 \
    --dataset="cityscapes" \
    --colormap_type="cityscapes" \
    --checkpoint_dir="/mrtstorage/users/rehman/experiments/c2e_encoding/m3l_c2e_b12_without_projections/" \
    --vis_logdir="/mrtstorage/users/rehman/experiments/tmp/vis/c2e/m3l_mage72NMS_with_single_projection"  \
    --dataset_dir="/mrtstorage/users/rehman/datasets/cityscapes/tfrecord_v3_c2e/" \





