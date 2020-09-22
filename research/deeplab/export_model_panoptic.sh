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
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --num_classes=19 \
    --decoder_output_stride="8,4" \
    --checkpoint_path="/mrtstorage/users/rehman/experiments/c2e_encoding/loss_v1/x65_b6_c0.5_further_pq_21/model.ckpt-42513" \
    --export_path="/mrtstorage/users/rehman/experiments/c2e_encoding/loss_v1/x65_b6_c0.5_further_pq_21/frozen_graph/frozen_graph.pb"  \




