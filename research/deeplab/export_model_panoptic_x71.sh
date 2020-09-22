set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"


# Set up the working directories.

python deeplab/export_model_panoptic.py \
    --model_variant="xception_71" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --num_classes=19 \
    --decoder_output_stride="8,4" \
    --checkpoint_path="/mrtstorage/users/rehman/experiments/old_models/x71_c2e_b4_c500_o20_co/model.ckpt-577" \
    --export_path="/mrtstorage/users/rehman/experiments/tmp/frozen_graph/x71frozen_graph/frozen_inference_graph_c2e_200.pb"  \




