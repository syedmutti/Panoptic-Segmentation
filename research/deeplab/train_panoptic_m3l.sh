set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# Set up the working directories.
EXP_DIR="/mrtstorage/users/rehman/experiments/c2e_encoding/loss_v1/m3l_small"
mkdir -p "${EXP_DIR}"
cp "${WORK_DIR}"/train_panoptic_m3l.sh "${EXP_DIR}"

python "${WORK_DIR}"/train_panoptic.py \
  --logtostderr \
  --train_split="train" \
  --model_variant="mobilenet_v3_small_seg" \
  --train_crop_size="769,769" \
  --train_batch_size=4\
  --decoder_output_stride="8,4" \
  --dataset="cityscapes" \
  --aspp_with_squeeze_and_excitation=1 \
  --image_se_uses_qsigmoid=1 \
  --image_pyramid=1 \
  --fine_tune_batch_norm=false \
  --save_summaries_secs=100 \
  --num_clones=1 \
  --initialize_last_layer=True \
  --last_layers_contain_logits_only=False \
  --training_number_of_steps=100000 \
  --tf_initial_checkpoint="/mrtstorage/users/rehman/experiments/c2e_encoding/loss_v1/m3l_small/model.ckpt" \
  --train_logdir="${EXP_DIR}" \
  --dataset_dir="/mrtstorage/users/rehman/datasets/cityscapes/tfrecord_v3_c2e"