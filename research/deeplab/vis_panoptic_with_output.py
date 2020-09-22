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


os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

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

def nms_centerpoint(image, kernel_size=16):
    test = np.array(image)
    center_dict = {}
    for h in range(test.shape[0] // kernel_size):
        for w in range(test.shape[1] // kernel_size - 1):

            local_region = test[h * kernel_size: (h * kernel_size) + kernel_size,
                           w * kernel_size:(w * kernel_size) + kernel_size]
            local_max = np.amax(local_region)
            if local_max > 12.75:
                idx = np.where(test == local_max)
                # Dictionary
                center_dict[local_max] = idx
    return center_dict

def process_panoptic_output(original_image, seg_map, center, offset, colormap_type):
    # Non-maxima suppression in Center Points
    max_center = np.amax(center)
    instance_center = np.multiply(np.divide(center, max_center), 255.0)
    center_dict = nms_centerpoint(instance_center, 32)
    center_points = []
    for index, key in enumerate(sorted(center_dict.keys(), reverse=True)):
        if index < 200:
            center_points.append(center_dict[key])
    instance_resolution = 16

    filtered_centers = center_points
    for center in center_points:
        for i, center_x in enumerate(center_points):
            if center_x != center:
                diff_center_h = np.square(np.subtract(center[0], center_x[0]))
                diff_center_w = np.square(np.subtract(center[1], center_x[1]))
                diff_dist = np.sqrt(diff_center_h + diff_center_w)
                if diff_dist < instance_resolution:
                    filtered_centers.pop(i)

    semantic_seg_array = seg_map
    unique_ids = np.unique(semantic_seg_array)
    instances_in_image = [unique_id for unique_id in unique_ids if (unique_id > 10 and unique_id <= 18)]
    panoptic_mask = np.zeros(np.array(seg_map).shape)
    # Extract panoptic mask from Semantic Segmentation

    for i in range(len(instances_in_image)):
        local_mask = semantic_seg_array == instances_in_image[i]
        panoptic_mask += local_mask

    panoptic_mask = panoptic_mask.astype('bool')

    red, green = np.dsplit(offset, 2)
    # Reverting values back and centering across zero.
    x_vectors = np.multiply(np.subtract(np.squeeze(red), 127.5), 2048/127.5)
    y_vectors = np.multiply(np.subtract(np.squeeze(green), 127.5), 2048/127.5)

    row, col = np.indices(np.squeeze(x_vectors).shape)
    stacked_offsets = np.zeros_like(x_vectors) + 5000

    for center in filtered_centers:
        h, w = center

        '''y_vectors[h[0] + 1:, :] *= -1
        x_vectors[:, w[0] + 1:] *= -1'''

        h_ = np.square(h[0] - np.add(row, y_vectors)) * panoptic_mask
        w_ = np.square(w[0] - np.add(col, x_vectors)) * panoptic_mask
        value = np.sqrt(h_ + w_)

        stacked_offsets = np.dstack((stacked_offsets, value))

    output = np.argmin(stacked_offsets, axis=-1)

    '''    panoptic_segmentation_scaled = panoptic_segmentation * (255 // len(np.unique(panoptic_segmentation)))
    inst_color = cv2.applyColorMap(panoptic_segmentation_scaled, cv2.COLORMAP_JET)
    panotpic_output = Image.blend(Image.fromarray(original_image), Image.fromarray(inst_color), 0.4)'''
    # For Creating boundries around instances
    # Add boundry to Image
    '''instance_boundry = np.zeros_like(seg_map)

    instances = np.unique(output)
    instances = np.delete(instances, 0)
    #print(instances)

  
    for index, i in enumerate(instances):
        local_instance_mask = output == i
        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(local_instance_mask.astype('uint8'), kernel, iterations=1)
        erosion = cv2.erode(local_instance_mask.astype('uint8'), kernel, iterations=1)
        boundry = (dilation - erosion) * 255
        instance_boundry += boundry

    colored_label = get_dataset_colormap.label_to_color_image(
        seg_map.astype('uint8'), colormap_type)
    colored_label = colored_label + np.dstack((instance_boundry, instance_boundry, instance_boundry))
    colored_label = Image.fromarray(colored_label.astype(dtype=np.uint8))

    panoptic_output = Image.blend(Image.fromarray(original_image), colored_label, 0.7)'''

    return np.array(output)


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


def _process_batch(sess, original_images, semantic_predictions, instance_predictions, regression_predictions, image_names,
                   image_heights, image_widths, image_id_offset, save_dir, instance_save_dir, regression_save_dir, panoptic_save_dir,
                   raw_save_dir, train_id_to_eval_id=None):
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
   image_names,
   image_heights,
   image_widths) = sess.run([original_images, semantic_predictions, instance_predictions, regression_predictions,
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

    '''if len(crop_regression_prediction.shape) == 3 and crop_regression_prediction.shape[2] == 2:
        red, green = np.dsplit(crop_regression_prediction, 3)
        blue = np.zeros_like(green)

        crop_regression_prediction = np.concatenate((red, green, blue), axis=2)

    else:
        raise ValueError('Input label y offset shape must be [height, width, 2].')
    #crop_regression_prediction = regression_predictions[:image_height, :image_width]'''

    panoptic_prediction = process_panoptic_output(original_image,
                                                  crop_semantic_prediction,
                                                  crop_instance_prediction,
                                                  crop_regression_prediction,
                                                  colormap_type=FLAGS.colormap_type)

    # Save image.
    save_annotation.save_annotation(
        original_image, save_dir, _IMAGE_FORMAT % (image_id_offset + i),
        add_colormap=False)

    # Save instance heatmap prediction.
    save_annotation.save_annotation(
        crop_instance_prediction, instance_save_dir,
        _PREDICTION_FORMAT % (image_id_offset + i), scale_values=True, add_colormap=False,
        colormap_type=FLAGS.colormap_type)

    # Save regression prediction.
    save_annotation.save_annotation_instance_regression(
        crop_regression_prediction, regression_save_dir,
        _PREDICTION_FORMAT % (image_id_offset + i), normalize_values=True, add_colormap=False,
        colormap_type=FLAGS.colormap_type)

    # Save prediction.
    save_annotation.save_annotation(
        crop_semantic_prediction, save_dir,
        _PREDICTION_FORMAT % (image_id_offset + i), add_colormap=True,
        colormap_type=FLAGS.colormap_type)

    # Save panoptic prediction.
    save_annotation.save_annotation_panoptic(
        panoptic_prediction, panoptic_save_dir,
        _PREDICTION_FORMAT % (image_id_offset + i), add_colormap=False,
        colormap_type=FLAGS.colormap_type)



    if FLAGS.also_save_raw_predictions:
      image_filename = os.path.basename(image_names[i])

      if train_id_to_eval_id is not None:
        crop_semantic_prediction = _convert_train_id_to_eval_id(
            crop_semantic_prediction,
            train_id_to_eval_id)
      save_annotation.save_annotation(
          crop_semantic_prediction, raw_save_dir, image_filename,
          add_colormap=False)


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

  regression_save_dir = os.path.join(FLAGS.vis_logdir, _OFFSET_PREDICTION_SAVE_FOLDER)
  tf.gfile.MakeDirs(regression_save_dir)

  panoptic_save_dir = os.path.join(FLAGS.vis_logdir, _PANOPTIC_PREDICTION_SAVE_FOLDER)
  tf.gfile.MakeDirs(panoptic_save_dir)

  raw_save_dir = os.path.join(
      FLAGS.vis_logdir, _RAW_SEMANTIC_PREDICTION_SAVE_FOLDER)
  tf.gfile.MakeDirs(raw_save_dir)

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
          predictions_regression,
          [0, 0, 0, 0],
          [1, original_image_shape[0], original_image_shape[1], 1])
      resized_shape = tf.to_int32([tf.squeeze(samples[common.HEIGHT]),
                                   tf.squeeze(samples[common.WIDTH]), 2])
      predictions_regression = tf.image.resize_images(predictions_regression,
                                 resized_shape,
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                 align_corners=True)

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
                         image_names=samples[common.IMAGE_NAME],
                         image_heights=samples[common.HEIGHT],
                         image_widths=samples[common.WIDTH],
                         image_id_offset=image_id_offset,
                         save_dir=save_dir,
                         instance_save_dir=instance_save_dir,
                         regression_save_dir=regression_save_dir,
                         panoptic_save_dir=panoptic_save_dir,
                         raw_save_dir=raw_save_dir,
                         train_id_to_eval_id=train_id_to_eval_id)
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
