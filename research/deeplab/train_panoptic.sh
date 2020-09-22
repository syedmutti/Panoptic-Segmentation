set -e
# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
# Set up the working directories.

EXP_DIR="/mrtstorage/users/rehman/experiments/c2e_encoding/loss_v1/x65_b1_c200_0.01_b3_9x9/further_full_resolution"
mkdir -p "${EXP_DIR}"
cp "${WORK_DIR}"/train_panoptic.sh "${EXP_DIR}"

# Train 150000 iterations.
NUM_ITERATIONS=50000
python "${WORK_DIR}"/train_panoptic.py \
    --logtostderr \
    --training_number_of_steps="${NUM_ITERATIONS}" \
    --train_split="train" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --train_crop_size="1024,1024" \
    --decoder_output_stride="8,4" \
    --train_batch_size=2 \
    --dataset="cityscapes" \
    --initialize_last_layer=True \
    --last_layers_contain_logits_only=False \
    --fine_tune_batch_norm=False \
    --save_summaries_secs=200 \
    --num_clones=2 \
    --tf_initial_checkpoint="/mrtstorage/users/rehman/experiments/c2e_encoding/loss_v1/x65_b1_c200_0.01_b3_9x9/model.ckpt-150000" \
    --train_logdir="${EXP_DIR}" \
    --dataset_dir="/mrtstorage/users/rehman/datasets/cityscapes/tfrecord_v4_c2e_9x9/"