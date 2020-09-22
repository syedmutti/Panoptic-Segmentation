# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Segmentation results visualization on a given set of images.

See model.py for more details and usage.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import os
import time
from mercurial.unionrepo import unionchangelog

import numpy as np
from six.moves import range
import tensorflow as tf
from tensorflow.contrib import quantize as contrib_quantize
from tensorflow.contrib import training as contrib_training
from deeplab import common
from deeplab import model_panoptic as model
from deeplab.datasets import data_generator_panoptic as data_generator
from deeplab.utils import save_annotation
from PIL import Image
import cv2
from deeplab.utils import get_dataset_colormap


os.environ["CUDA_VISIBLE_DEVICES"]="0"

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

# Settings for log directories.

flags.DEFINE_string('vis_logdir', None, 'Where to write the event logs.')

flags.DEFINE_string('checkpoint_dir', None, 'Directory of model checkpoints.')

# Settings for visualizing the model.

flags.DEFINE_integer('vis_batch_size', 1,
                     'The number of images in each batch during evaluation.')

flags.DEFINE_list('vis_crop_size', '513,513',
                  'Crop size [height, width] for visualization.')

flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                     'How often (in seconds) to run evaluation.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Change to [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] for multi-scale test.
flags.DEFINE_multi_float('eval_scales', [1.0],
                         'The scales to resize images for evaluation.')

# Change to True for adding flipped images during test.
flags.DEFINE_bool('add_flipped_images', False,
                  'Add flipped images for evaluation or not.')

flags.DEFINE_integer(
    'quantize_delay_step', -1,
    'Steps to start quantized training. If < 0, will not quantize model.')

# Dataset settings.

flags.DEFINE_string('dataset', 'pascal_voc_seg',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('vis_split', 'val',
                    'Which split of the dataset used for visualizing results')

flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')

flags.DEFINE_enum('colormap_type', 'pascal', ['pascal', 'cityscapes', 'ade20k'],
                  'Visualization colormap type.')

flags.DEFINE_boolean('also_save_raw_predictions', False,
                     'Also save raw predictions.')

flags.DEFINE_integer('max_number_of_iterations', 0,
                     'Maximum number of visualization iterations. Will loop '
                     'indefinitely upon nonpositive values.')

# The folder where semantic segmentation predictions are saved.
_SEMANTIC_PREDICTION_SAVE_FOLDER = 'segmentation_results'

# The folder where instance center predictions are saved.
_INSTANCE_PREDICTION_SAVE_FOLDER = 'instance_center_results'

# The folder where panoptic predictions are saved.
_PANOPTIC_PREDICTION_SAVE_FOLDER = 'panoptic_segmentation_results'

# The folder where instance regression predictions are saved.
_OFFSET_PREDICTION_SAVE_FOLDER = 'instance_offset_results'

_INSTANCE_SEG_SAVE_FOLDER = 'instance_seg_results'

# The folder where raw semantic segmentation predictions are saved.
_RAW_SEMANTIC_PREDICTION_SAVE_FOLDER = 'raw_segmentation_results'

# The format to save image.
_IMAGE_FORMAT = '%06d_image'

# The format to save prediction
_PREDICTION_FORMAT = '%06d_prediction'

# To evaluate Cityscapes results on the evaluation server, the labels used
# during training should be mapped to the labels for evaluation.
_CITYSCAPES_TRAIN_ID_TO_EVAL_ID = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                                   23, 24, 25, 26, 27, 28, 31, 32, 33]


def generate_instance_segmentation(predictions_semantic, predictions_center_points, predictions_offset_vectors):

    predictions_semantic = tf.squeeze(predictions_semantic)

    intermediate_panoptic_mask = tf.greater_equal(predictions_semantic, 11)
    predictions_semantic = predictions_semantic * tf.cast(intermediate_panoptic_mask, tf.int64)
    predictions_semantic = predictions_semantic * tf.cast(tf.less_equal(predictions_semantic, 18), tf.int64)

    panoptic_mask = tf.squeeze(tf.cast(tf.not_equal(predictions_semantic, 0), tf.float32))
    predictions_offset_vectors = tf.squeeze(predictions_offset_vectors)


    predictions_center_points = tf.multiply(predictions_center_points / tf.reduce_max(predictions_center_points), 255)
    keep_mask = tf.greater_equal(predictions_center_points, 100)
    predictions_center_points = predictions_center_points * tf.cast(keep_mask, tf.float32)
    predictions_center_points = predictions_center_points * tf.expand_dims(tf.expand_dims(panoptic_mask, 0), -1)

    x_vectors, y_vectors = tf.split(predictions_offset_vectors, num_or_size_splits=2, axis=2)

    x_vectors = tf.squeeze(x_vectors) * panoptic_mask
    y_vectors = tf.squeeze(y_vectors) * panoptic_mask


    squeezed_heatmap = tf.squeeze(predictions_center_points)
    kernel_size = 72
    output = tf.nn.max_pool_with_argmax(
        predictions_center_points,
        [1, kernel_size, kernel_size, 1],
        [1, kernel_size, kernel_size, 1],
        "SAME", include_batch_in_index=False)

    squeezed_heatmap = tf.squeeze(predictions_center_points)
    values = tf.squeeze(output.output)
    indices = tf.squeeze(output.argmax)

    values_reshaped = tf.reshape(values, shape=[-1])
    indices_reshaped = tf.reshape(indices, shape=[-1])

    top_values, top_indices = tf.math.top_k(values_reshaped,
                                            k=200,
                                            sorted=True,
                                            name=None)

    long_indices = tf.gather(indices_reshaped, top_indices)

    unraveled_indices = tf.unravel_index(
        indices=long_indices, dims=[tf.shape(squeezed_heatmap)[0], tf.shape(squeezed_heatmap)[1]])

    unraveled_indices = tf.squeeze(unraveled_indices)

    row, col = tf.split(unraveled_indices, num_or_size_splits=2, axis=0)

    row = tf.squeeze(row)
    col = tf.squeeze(col)

    unraveled_indices = tf.stack([row, col], axis=1)
    unraveled_indices = tf.cast(unraveled_indices, tf.float32)

    rows, columns = np.indices((1024, 2048))
    rows_indices = tf.constant(rows)
    cols_indices = tf.constant(columns)

    rows_indices = tf.cast(tf.squeeze(rows_indices), tf.float32)
    cols_indices = tf.cast(tf.squeeze(cols_indices), tf.float32)

    def true_fn(index):

        row, column = tf.split(index, num_or_size_splits=2, axis=0)

        # Inverting Offset Vectors to preform instance regression
        inversion_mask_row = tf.cast(tf.greater(rows_indices, row), tf.float32) * -1
        not_inversion_mask_row = tf.cast(tf.less_equal(rows_indices, row), tf.float32)
        rows_mask = tf.add(inversion_mask_row, not_inversion_mask_row)

        inversion_mask_col = tf.cast(tf.greater(cols_indices, column), tf.float32) * -1
        not_inversion_mask_col = tf.cast(tf.less_equal(cols_indices, column), tf.float32)
        cols_mask = tf.add(inversion_mask_col, not_inversion_mask_col)

        # Instance Regressions
        h_ = tf.square(tf.subtract(row, tf.add(rows_indices, y_vectors * rows_mask))) * panoptic_mask
        w_ = tf.square(tf.subtract(column, tf.add(cols_indices, x_vectors * cols_mask))) * panoptic_mask
        distance_map = tf.square(tf.sqrt(h_ + w_))   # (Adding to increase the values around the instances to seperate background)

        return distance_map

    def false_fn(x):
        return tf.zeros(tf.shape(squeezed_heatmap)) + 3000

    def myfunc(x):
        pred = tf.gather_nd(squeezed_heatmap, tf.cast(x, dtype=tf.int32))
        result = tf.cond(tf.greater(pred, 0), lambda: true_fn(x), lambda: false_fn(x))

        return result

    output = tf.map_fn(lambda x: myfunc(x), unraveled_indices, dtype=tf.float32)

    background_zeros = tf.zeros_like(squeezed_heatmap) + 3000
    background_zeros = tf.expand_dims(background_zeros * panoptic_mask, 0)
    output = tf.concat([background_zeros, output], 0)
    instance_prediction = tf.argmin(output, axis=None, output_type=tf.dtypes.int32)

    return instance_prediction


def _convert_train_id_to_eval_id(prediction, train_id_to_eval_id):
  """Converts the predicted label for evaluation.

  There are cases where the training labels are not equal to the evaluation
  labels. This function is used to perform the conversion so that we could
  evaluate the results on the evaluation server.

  Args:
    prediction: Semantic segmentation prediction.
    train_id_to_eval_id: A list mapping from train id to evaluation id.

  Returns:
    Semantic segmentation prediction whose labels have been changed.
  """
  converted_prediction = prediction.copy()
  for train_id, eval_id in enumerate(train_id_to_eval_id):
    converted_prediction[prediction == train_id] = eval_id

  return converted_prediction


def _process_batch(sess, original_images, semantic_predictions, instance_predictions, regression_predictions, panoptic_prediction, image_names,
                   image_heights, image_widths, image_id_offset, save_dir, instance_save_dir, instance_seg_save_dir, panoptic_save_dir,
                   instance_offset_save_dir):
  """Evaluates one single batch qualitatively.

  Args:
    sess: TensorFlow session.
    original_images: One batch of original images.
    semantic_predictions: One batch of semantic segmentation predictions.
    instance_predictions: One batch of instance predictions.
    image_names: Image names.
    image_heights: Image heights.
    image_widths: Image widths.
    image_id_offset: Image id offset for indexing images.
    save_dir: The directory where the predictions will be saved.
    instance_save_dir : The directory where the instance predictions will be saved.
    raw_save_dir: The directory where the raw predictions will be saved.
    train_id_to_eval_id: A list mapping from train id to eval id.
  """
  (original_images,
   semantic_predictions,
   instance_predictions,
   regression_predictions,
   instance_segmentation,
   image_names,
   image_heights,
   image_widths) = sess.run([original_images, semantic_predictions, instance_predictions, regression_predictions, panoptic_prediction,
                             image_names, image_heights, image_widths])

  num_image = semantic_predictions.shape[0]
  for i in range(num_image):
    image_height = np.squeeze(image_heights[i])
    image_width = np.squeeze(image_widths[i])
    original_image = np.squeeze(original_images[i])
    semantic_prediction = np.squeeze(semantic_predictions[i])
    instance_predictions = np.squeeze(instance_predictions[i])
    regression_predictions = np.squeeze(regression_predictions[i])


    crop_semantic_prediction = semantic_prediction[:image_height, :image_width]
    crop_instance_prediction = instance_predictions[:image_height, :image_width]
    crop_regression_prediction = regression_predictions[:image_height, :image_width, :]

    instance_segmentation = np.squeeze(instance_segmentation)
    unique_elements = np.unique(instance_segmentation)

    instance_segmentation_scaled = np.array(instance_segmentation) * (255//len(unique_elements))


    #################### Heatmaps ############################

    crop_instance_prediction = np.multiply(crop_instance_prediction/np.amax(crop_instance_prediction), 255)
    #crop_instance_prediction_mask = np.greater_equal(crop_instance_prediction, 100)
    crop_instance_prediction_mask = np.greater_equal(crop_instance_prediction, 10)
    crop_instance_prediction = crop_instance_prediction * crop_instance_prediction_mask

    mask_single_channel = np.not_equal(crop_instance_prediction, 0)
    mask_rgb = np.dstack((mask_single_channel, mask_single_channel, mask_single_channel))

    instance_heatmap = cv2.applyColorMap(crop_instance_prediction.astype('uint8'), cv2.COLORMAP_HSV)
    #instance_heatmap = Image.blend(Image.fromarray(original_image), Image.fromarray(instance_heatmap), 0.2)
    instance_heatmap = Image.fromarray(np.where(mask_rgb, instance_heatmap, original_image))

    ##################### Offset Vectors Prediction #########################

    red, green = np.dsplit(crop_regression_prediction, 2)
    blue = np.zeros_like(red)
    crop_regression_prediction = np.dstack((red, green, blue))
    crop_regression_prediction = np.multiply(np.divide(crop_regression_prediction, np.amax(crop_regression_prediction)), 255)
    ##########  VIS PANOPTIC OUTPUT ##################

    inst_color = cv2.applyColorMap(instance_segmentation_scaled.astype('uint8'), cv2.COLORMAP_JET)
    instance_segmentation_coloured = Image.blend(Image.fromarray(original_image), Image.fromarray(inst_color), 0.4)

    # For Creating boundries around instances
    # Add boundry to Image
    colormap_type = FLAGS.colormap_type
    instance_boundry = np.zeros_like(semantic_prediction)
    instances = np.delete(unique_elements, 0)

    for index, i in enumerate(instances):
        local_instance_mask = instance_segmentation == i
        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(local_instance_mask.astype('uint8'), kernel, iterations=1)
        erosion = cv2.erode(local_instance_mask.astype('uint8'), kernel, iterations=1)
        boundry = (dilation - erosion) * 255
        instance_boundry += boundry

    colored_label = get_dataset_colormap.label_to_color_image(
        semantic_prediction.astype('uint8'), colormap_type)
    colored_label = colored_label + np.dstack((instance_boundry, instance_boundry, instance_boundry))
    colored_label = Image.fromarray(colored_label.astype(dtype=np.uint8))

    panoptic_output = Image.blend(Image.fromarray(original_image), colored_label, 0.7)


    # Save image.
    save_annotation.save_annotation(
        original_image, save_dir, _IMAGE_FORMAT % (image_id_offset + i),
        add_colormap=False)

    # Save instance heatmap prediction.
    save_annotation.save_annotation_heatmaps(
        instance_heatmap, instance_save_dir,
        _PREDICTION_FORMAT % (image_id_offset + i), scale_values=False, add_colormap=False,
        colormap_type=FLAGS.colormap_type)

    # Save regression prediction.
    save_annotation.save_annotation_instance_segmentation(
        instance_segmentation_coloured, instance_seg_save_dir ,
        _PREDICTION_FORMAT % (image_id_offset + i), normalize_values=False, add_colormap=False,
        colormap_type=FLAGS.colormap_type)

    # Save prediction.
    save_annotation.save_annotation_overlayed(
        crop_semantic_prediction, original_image, save_dir,
        _PREDICTION_FORMAT % (image_id_offset + i), add_colormap=True,
        colormap_type=FLAGS.colormap_type)

    # Save panoptic prediction.
    save_annotation.save_annotation_panoptic(
        panoptic_output, panoptic_save_dir,
        _PREDICTION_FORMAT % (image_id_offset + i), add_colormap=False,
        colormap_type=FLAGS.colormap_type)

    # Save instance_segmentation prediction.
    save_annotation.save_annotation_offset_vectors(
        crop_regression_prediction, instance_offset_save_dir,
        _PREDICTION_FORMAT % (image_id_offset + i))



def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  # Get dataset-dependent information.
  dataset = data_generator.Dataset(
      dataset_name=FLAGS.dataset,
      split_name=FLAGS.vis_split,
      dataset_dir=FLAGS.dataset_dir,
      batch_size=FLAGS.vis_batch_size,
      crop_size=[int(sz) for sz in FLAGS.vis_crop_size],
      min_resize_value=FLAGS.min_resize_value,
      max_resize_value=FLAGS.max_resize_value,
      resize_factor=FLAGS.resize_factor,
      model_variant=FLAGS.model_variant,
      is_training=False,
      should_shuffle=False,
      should_repeat=False)

  train_id_to_eval_id = None
  if dataset.dataset_name == data_generator.get_cityscapes_dataset_name():
    tf.logging.info('Cityscapes requires converting train_id to eval_id.')
    train_id_to_eval_id = _CITYSCAPES_TRAIN_ID_TO_EVAL_ID

  # Prepare for visualization.
  tf.gfile.MakeDirs(FLAGS.vis_logdir)

  save_dir = os.path.join(FLAGS.vis_logdir, _SEMANTIC_PREDICTION_SAVE_FOLDER)
  tf.gfile.MakeDirs(save_dir)

  instance_save_dir = os.path.join(FLAGS.vis_logdir, _INSTANCE_PREDICTION_SAVE_FOLDER)
  tf.gfile.MakeDirs(instance_save_dir)

  instance_seg_save_dir = os.path.join(FLAGS.vis_logdir, _INSTANCE_SEG_SAVE_FOLDER)
  tf.gfile.MakeDirs(instance_seg_save_dir)

  instance_offset_save_dir = os.path.join(FLAGS.vis_logdir, _OFFSET_PREDICTION_SAVE_FOLDER)
  tf.gfile.MakeDirs(instance_offset_save_dir)

  panoptic_save_dir = os.path.join(FLAGS.vis_logdir, _PANOPTIC_PREDICTION_SAVE_FOLDER)
  tf.gfile.MakeDirs(panoptic_save_dir)


  tf.logging.info('Visualizing on %s set', FLAGS.vis_split)

  with tf.Graph().as_default():
    samples = dataset.get_one_shot_iterator().get_next()

    model_options = common.ModelOptions(
        outputs_to_num_classes={
          common.OUTPUT_TYPE: dataset.num_of_classes,
          common.INSTANCE: 1,
            common.OFFSET: 2

      },
        crop_size=[int(sz) for sz in FLAGS.vis_crop_size],
        atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.output_stride)

    if tuple(FLAGS.eval_scales) == (1.0,):
      tf.logging.info('Performing single-scale test.')
      predictions = model.predict_labels(
          samples[common.IMAGE],
          model_options=model_options,
          image_pyramid=FLAGS.image_pyramid)
    else:
      tf.logging.info('Performing multi-scale test.')
      if FLAGS.quantize_delay_step >= 0:
        raise ValueError(
            'Quantize mode is not supported with multi-scale test.')
      predictions = model.predict_labels_multi_scale(
          samples[common.IMAGE],
          model_options=model_options,
          eval_scales=FLAGS.eval_scales,
          add_flipped_images=FLAGS.add_flipped_images)

    predictions_semantic = predictions[common.OUTPUT_TYPE]
    predictions_instance = predictions[common.INSTANCE]
    predictions_regression = predictions[common.OFFSET]

    if FLAGS.min_resize_value and FLAGS.max_resize_value:
      # Only support batch_size = 1, since we assume the dimensions of original
      # image after tf.squeeze is [height, width, 3].
      assert FLAGS.vis_batch_size == 1

      # Reverse the resizing and padding operations performed in preprocessing.
      # First, we slice the valid regions (i.e., remove padded region) and then
      # we resize the predictions back.
      original_image = tf.squeeze(samples[common.ORIGINAL_IMAGE])
      original_image_shape = tf.shape(original_image)
      predictions_semantic = tf.slice(
          predictions_semantic,
          [0, 0, 0],
          [1, original_image_shape[0], original_image_shape[1]])
      resized_shape = tf.to_int32([tf.squeeze(samples[common.HEIGHT]),
                                   tf.squeeze(samples[common.WIDTH])])
      predictions_semantic = tf.squeeze(
          tf.image.resize_images(tf.expand_dims(predictions_semantic, 3),
                                 resized_shape,
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                 align_corners=True), 3)
      ############################### POST PROCESSING LOGITS FROM INSTANCE CENTER #####################
      predictions_instance = tf.slice(
          predictions_instance,
          [0, 0, 0],
          [1, original_image_shape[0], original_image_shape[1]])
      resized_shape = tf.to_int32([tf.squeeze(samples[common.HEIGHT]),
                                   tf.squeeze(samples[common.WIDTH])])
      predictions_instance = tf.squeeze(
          tf.image.resize_images(tf.expand_dims(predictions_instance, 3),
                                 resized_shape,
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                 align_corners=True), 3)

      ############################### POST PROCESSING LOGITS FROM INSTANCE REGRESSION #####################
      predictions_regression = tf.slice(
          predictions_regression
          [0, 0, 0, 0],
          [1, original_image_shape[0], original_image_shape[1], 1])
      resized_shape = tf.to_int32([tf.squeeze(samples[common.HEIGHT]),
                                   tf.squeeze(samples[common.WIDTH]), 2])
      predictions_regression = tf.image.resize_images(predictions_regression,
                                 resized_shape,
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                 align_corners=True)
    ###########################################################################################################


    panoptic_output = generate_instance_segmentation(predictions_semantic, predictions_instance, predictions_regression)

      ###########################################################################################################

    tf.train.get_or_create_global_step()
    if FLAGS.quantize_delay_step >= 0:
      contrib_quantize.create_eval_graph()

    num_iteration = 0
    max_num_iteration = FLAGS.max_number_of_iterations

    checkpoints_iterator = contrib_training.checkpoints_iterator(
        FLAGS.checkpoint_dir, min_interval_secs=FLAGS.eval_interval_secs)
    for checkpoint_path in checkpoints_iterator:
      num_iteration += 1
      tf.logging.info(
          'Starting visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                       time.gmtime()))
      tf.logging.info('Visualizing with model %s', checkpoint_path)

      scaffold = tf.train.Scaffold(init_op=tf.global_variables_initializer())
      session_creator = tf.train.ChiefSessionCreator(
          scaffold=scaffold,
          master=FLAGS.master,
          checkpoint_filename_with_path=checkpoint_path)
      with tf.train.MonitoredSession(
          session_creator=session_creator, hooks=None) as sess:
        batch = 0
        image_id_offset = 0

        while not sess.should_stop():
          tf.logging.info('Visualizing batch %d', batch + 1)
          _process_batch(sess=sess,
                         original_images=samples[common.ORIGINAL_IMAGE],
                         semantic_predictions=predictions_semantic,
                         instance_predictions=predictions_instance,
                         regression_predictions=predictions_regression,
                         panoptic_prediction=panoptic_output,
                         image_names=samples[common.IMAGE_NAME],
                         image_heights=samples[common.HEIGHT],
                         image_widths=samples[common.WIDTH],
                         image_id_offset=image_id_offset,
                         save_dir=save_dir,
                         instance_save_dir=instance_save_dir,
                         instance_seg_save_dir=instance_seg_save_dir,
                         panoptic_save_dir=panoptic_save_dir,
                         instance_offset_save_dir=instance_offset_save_dir)

          image_id_offset += FLAGS.vis_batch_size
          batch += 1

      tf.logging.info(
          'Finished visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                       time.gmtime()))
      if max_num_iteration > 0 and num_iteration >= max_num_iteration:
        break

if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_dir')
  flags.mark_flag_as_required('vis_logdir')
  flags.mark_flag_as_required('dataset_dir')
  tf.app.run()