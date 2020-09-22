set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"


# Set up the working directories.
# mobilenet_v3_large_seg
python deeplab/export_model_panoptic.py \
    --model_variant="mobilenet_v3_small_seg" \
    --aspp_with_squeeze_and_excitation=1 \
    --image_se_uses_qsigmoid=1 \
    --dataset="cityscapes" \
    --num_classes=19 \
    --decoder_output_stride="8,4" \
    --checkpoint_path="/mrtstorage/users/rehman/experiments/c2e_encoding/loss_v1/m3l_small/model.ckpt-0" \
    --export_path="/mrtstorage/users/rehman/experiments/tmp/frozen_graph/c2e_m3_small/frozen_inference_graph_c2e_50.pb"  \




