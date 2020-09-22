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
"""Evaluation script for the DeepLab model.

See model.py for more details and usage.
"""

import numpy as np
import six
import tensorflow as tf
from PIL import Image
from tensorflow.contrib import metrics as contrib_metrics
from tensorflow.contrib import quantize as contrib_quantize
from tensorflow.contrib import tfprof as contrib_tfprof
from tensorflow.contrib import training as contrib_training
from tensorflow.contrib import slim as contrib_slim
from deeplab import common
import os
from deeplab import model_panoptic as model
from deeplab.evaluation import streaming_metrics
from deeplab.datasets import data_generator_panoptic as data_generator

os.environ["CUDA_VISIBLE_DEVICES"]="2"

slim = contrib_slim

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

# Settings for log directories.

flags.DEFINE_string('eval_logdir', None, 'Where to write the event logs.')

flags.DEFINE_string('checkpoint_dir', None, 'Directory of model checkpoints.')

# Settings for evaluating the model.

flags.DEFINE_integer('eval_batch_size', 1,
                     'The number of images in each batch during evaluation.')

flags.DEFINE_list('eval_crop_size', '513,513',
                  'Image crop size [height, width] for evaluation.')

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

flags.DEFINE_string('eval_split', 'val',
                    'Which split of the dataset used for evaluation')

flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')

flags.DEFINE_integer('max_number_of_evaluations', 0,
                     'Maximum number of eval iterations. Will loop '
                     'indefinitely upon nonpositive values.')


def generate_instance_segmentation(predictions_semantic, predictions_center_points, predictions_offset_vectors, instance_label_mask):

    predictions_semantic = tf.squeeze(predictions_semantic)
    intermediate_panoptic_mask = tf.greater_equal(predictions_semantic, 11)
    predictions_semantic = predictions_semantic * tf.cast(intermediate_panoptic_mask, tf.int64)
    predictions_semantic = predictions_semantic * tf.cast(tf.less_equal(predictions_semantic, 18), tf.int64)

    panoptic_mask = tf.cast(tf.not_equal(predictions_semantic, 0), tf.float32)
    panoptic_mask = panoptic_mask * tf.cast(instance_label_mask, tf.float32)

    # Masking Heatmap with Instance Predictions
    predictions_center_points = tf.multiply(predictions_center_points / tf.reduce_max(predictions_center_points), 255)
    keep_mask = tf.greater_equal(predictions_center_points, 127.5)
    predictions_center_points = predictions_center_points * tf.cast(keep_mask, tf.float32)
    predictions_center_points = predictions_center_points * tf.expand_dims(tf.expand_dims(panoptic_mask, 0), -1)

    # Converting to original Values and Masking
    x_vectors, y_vectors = tf.split(tf.squeeze(predictions_offset_vectors), num_or_size_splits=2, axis=-1)
    x_vectors = tf.squeeze(x_vectors) * panoptic_mask
    y_vectors = tf.squeeze(y_vectors) * panoptic_mask

    squeezed_heatmap = tf.squeeze(predictions_center_points)
    kernel_size = 72
    output = tf.nn.max_pool_with_argmax(
        predictions_center_points,
        [1, kernel_size, kernel_size, 1],
        [1, kernel_size, kernel_size, 1],
        "SAME", include_batch_in_index=False)  # (2,32,64,1)

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
        distance_map = tf.square(
            tf.sqrt(h_ + w_))  # (Adding to increase the values around the instances to seperate background)

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


'''def process_panoptic_output(seg_map, center_pred, instance_indices, values_top_k, offset):

    values_top_k = np.array(values_top_k)
    instance_indices = np.array(instance_indices)
    original_indices = np.unravel_index(instance_indices[values_top_k[1]], seg_map.shape)

    rows, col = original_indices
    center_points = []
    for index, row in enumerate(rows):
        if center_pred[row, col[index]] == 0:
            break
        center_points.append((row, col[index]))
    print('Number of Centers :', len(center_points))

    instance_resolution = 16
    filtered_centers = center_points

    print('Filtering center Points')

    for center in center_points:
        for i, center_x in enumerate(center_points):
            if center_x != center:
                diff_center_h = np.square(np.subtract(center[0], center_x[0]))
                diff_center_w = np.square(np.subtract(center[1], center_x[1]))
                diff_dist = np.sqrt(diff_center_h + diff_center_w)
                if diff_dist < instance_resolution:
                    filtered_centers.pop(i)
    print('Center Points after filtering : ', len(filtered_centers))

    semantic_seg_array = seg_map
    unique_ids = np.unique(semantic_seg_array)
    instances_in_image = [unique_id for unique_id in unique_ids if (unique_id > 10 and unique_id <= 18)]
    panoptic_mask = np.zeros(np.array(seg_map).shape)

    # Extract panoptic mask from Semantic Segmentation
    for i in range(len(instances_in_image)):
        local_mask = semantic_seg_array == instances_in_image[i]
        panoptic_mask += local_mask
    print('generated instance mask')
    panoptic_mask = panoptic_mask.astype('bool')

    red, green = np.dsplit(offset, 2)
    x_vectors = np.squeeze(red)
    y_vectors = np.squeeze(green)

    row, col = np.indices(np.squeeze(x_vectors).shape)
    stacked_offsets = np.zeros_like(y_vectors) + 5000

    for center in center_points:
        h, w = center

        x_vectors = np.multiply(np.subtract(x_vectors, 127.5), 2048/127.5)
        y_vectors = np.multiply(np.subtract(y_vectors, 127.5), 2048/127.5)

        y_vectors[h[0] + 1:, :] *= -1
        x_vectors[:, w[0] + 1:] *= -1

        h_ = np.square(h - np.add(row, y_vectors)) * panoptic_mask
        w_ = np.square(w - np.add(col, x_vectors)) * panoptic_mask
        value = np.sqrt(h_ + w_)

        stacked_offsets = np.dstack((stacked_offsets, value))

    output = np.argmin(stacked_offsets, axis=-1)

    return np.array(output)

def instance_output(semantic_predictions, instance_center, instance_indices, top_k, regression_predictions):

    num_image = semantic_predictions.shape[0]
    for i in range(num_image):

        semantic_prediction = np.squeeze(semantic_predictions[i])
        instance_center_prediction = np.squeeze(instance_center[i])
        regression_predictions = np.squeeze(regression_predictions[i])

        crop_semantic_prediction = semantic_prediction
        crop_instance_center = instance_center_prediction
        crop_regression_prediction = regression_predictions


        instnace_prediction = process_panoptic_output(crop_semantic_prediction,
                                                      crop_instance_center,
                                                      instance_indices,
                                                      top_k,
                                                      crop_regression_prediction)

        return crop_semantic_prediction, instnace_prediction'''

def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  dataset = data_generator.Dataset(
      dataset_name=FLAGS.dataset,
      split_name=FLAGS.eval_split,
      dataset_dir=FLAGS.dataset_dir,
      batch_size=FLAGS.eval_batch_size,
      crop_size=[int(sz) for sz in FLAGS.eval_crop_size],
      min_resize_value=FLAGS.min_resize_value,
      max_resize_value=FLAGS.max_resize_value,
      resize_factor=FLAGS.resize_factor,
      model_variant=FLAGS.model_variant,
      num_readers=2,
      is_training=False,
      should_shuffle=False,
      should_repeat=False)

  tf.gfile.MakeDirs(FLAGS.eval_logdir)
  tf.logging.info('Evaluating on %s set', FLAGS.eval_split)

  with tf.Graph().as_default():
    samples = dataset.get_one_shot_iterator().get_next()

    model_options = common.ModelOptions(
        outputs_to_num_classes={
             common.OUTPUT_TYPE: dataset.num_of_classes,
             common.INSTANCE: 1,
             common.OFFSET: 2},
        crop_size=[int(sz) for sz in FLAGS.eval_crop_size],
        atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.output_stride)

    # Set shape in order for tf.contrib.tfprof.model_analyzer to work properly.
    samples[common.IMAGE].set_shape(
        [FLAGS.eval_batch_size,
         int(FLAGS.eval_crop_size[0]),
         int(FLAGS.eval_crop_size[1]),
         3])

    if tuple(FLAGS.eval_scales) == (1.0,):
      tf.logging.info('Performing single-scale test.')
      predictions = model.predict_labels(samples[common.IMAGE], model_options,
                                         image_pyramid=FLAGS.image_pyramid)
    else:
      tf.logging.info('Performing multi-scale test.')
      if FLAGS.quantize_delay_step >= 0:
        raise ValueError(
            'Quantize mode is not supported with multi-scale test.')

    predictions_semantic = predictions[common.OUTPUT_TYPE]
    predictions_center_points = predictions[common.INSTANCE]
    predictions_offset_vectors = predictions[common.OFFSET]

    instance_label = tf.squeeze(samples[common.LABEL_INSTANCE_IDS][0])

    # Removing Instances smaller than 2048 pixels
    def filter_smallobj(instance_pre):
        unique_ids, pixels_count = np.unique(instance_pre, return_counts=True)
        truth_value = pixels_count > 3048
        unique_ids *= truth_value
        big_instances = np.unique(unique_ids)
        mask_for_instances = np.zeros_like(instance_pre)

        for instance in big_instances:
            local_mask = instance_pre == instance
            mask_for_instances += local_mask
        Big_instances_result = instance_pre * mask_for_instances
        return Big_instances_result

    input_args = [instance_label]
    instance_label_processed = tf.numpy_function(func=filter_smallobj, inp=input_args, Tout=[tf.uint8])
    instance_label_mask = tf.not_equal(tf.squeeze(instance_label_processed), 0)

    # tf Non-maxima Suppression
    # Pooling based NMS for Pooling Instance Centers
    # Filtering predictions that are less than 0.1
    instance_prediction = generate_instance_segmentation(predictions_semantic, predictions_center_points, predictions_offset_vectors, instance_label_mask)

    category_prediction = tf.squeeze(predictions_semantic)

    category_label = tf.squeeze(samples[common.LABEL][0])
    instance_label = tf.squeeze(samples[common.LABEL_INSTANCE_IDS][0])
    instance_prediction = instance_prediction * tf.cast(instance_label_mask, tf.int32)

    # Define the evaluation metric.
    metric_map = {}
    metric_map['panoptic_quality'] = streaming_metrics.streaming_panoptic_quality(
        category_label,
        instance_label_processed,
        category_prediction,
        instance_prediction,
        num_classes=19,
        max_instances_per_category=256,
        ignored_label=255,
        offset=256 * 256)
    metric_map['parsing_covering'] = streaming_metrics.streaming_parsing_covering(
        category_label,
        instance_label,
        category_prediction,
        instance_prediction,
        num_classes=19,
        max_instances_per_category=256,
        ignored_label=255,
        offset=256 * 256,
        normalize_by_image_size=True)
    metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map(
        metric_map)

    summary_ops = []
    for metric_name, metric_value in metrics_to_values.iteritems():
        if metric_name == 'panoptic_quality':
            [pq, sq, rq, total_tp, total_fn, total_fp] = tf.unstack(
                metric_value, 6, axis=0)
            panoptic_metrics = {
                # Panoptic quality.
                'pq': pq,
                # Segmentation quality.
                'sq': sq,
                # Recognition quality.
                'rq': rq,
                # Total true positives.
                'total_tp': total_tp,
                # Total false negatives.
                'total_fn': total_fn,
                # Total false positives.
                'total_fp': total_fp,
            }
            # Find the valid classes that will be used for evaluation. We will
            # ignore the `ignore_label` class and other classes which have (tp + fn
            # + fp) equal to 0.
            valid_classes = tf.logical_and(
                tf.not_equal(tf.range(0, dataset.num_of_classes), dataset.ignore_label),
                tf.not_equal(total_tp + total_fn + total_fp, 0))
            for target_metric, target_value in panoptic_metrics.iteritems():
                output_metric_name = '{}_{}'.format(metric_name, target_metric)
                op = tf.summary.scalar(
                    output_metric_name,
                    tf.reduce_mean(tf.boolean_mask(target_value, valid_classes)))
                op = tf.Print(op, [target_value], output_metric_name + '_classwise: ',
                              summarize=dataset.num_of_classes)
                op = tf.Print(
                    op,
                    [tf.reduce_mean(tf.boolean_mask(target_value, valid_classes))],
                    output_metric_name + '_mean: ',
                    summarize=1)
                summary_ops.append(op)
        elif metric_name == 'parsing_covering':
            [per_class_covering,
             total_per_class_weighted_ious,
             total_per_class_gt_areas] = tf.unstack(metric_value, 3, axis=0)
            # Find the valid classes that will be used for evaluation. We will
            # ignore the `void_label` class and other classes which have
            # total_per_class_weighted_ious + total_per_class_gt_areas equal to 0.
            valid_classes = tf.logical_and(
                tf.not_equal(tf.range(0, dataset.num_of_classes), dataset.ignore_label),
                tf.not_equal(
                    total_per_class_weighted_ious + total_per_class_gt_areas, 0))
            op = tf.summary.scalar(
                metric_name,
                tf.reduce_mean(tf.boolean_mask(per_class_covering, valid_classes)))
            op = tf.Print(op, [per_class_covering], metric_name + '_classwise: ',
                          summarize=dataset.num_of_classes)
            op = tf.Print(
                op,
                [tf.reduce_mean(
                    tf.boolean_mask(per_class_covering, valid_classes))],
                metric_name + '_mean: ',
                summarize=1)
            summary_ops.append(op)
        else:
            raise ValueError('The metric_name "%s" is not supported.' % metric_name)


    num_eval_iters = None
    if FLAGS.max_number_of_evaluations > 0:
      num_eval_iters = FLAGS.max_number_of_evaluations

    if FLAGS.quantize_delay_step >= 0:
      contrib_quantize.create_eval_graph()

    contrib_tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=contrib_tfprof.model_analyzer
        .TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    contrib_tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=contrib_tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

    metric_values = slim.evaluation.evaluation_loop(
        master=FLAGS.master,
        checkpoint_dir=FLAGS.checkpoint_dir,
        logdir=FLAGS.eval_logdir,
        num_evals=500,
        eval_op=metrics_to_updates.values(),
        final_op=metrics_to_values.values(),
        summary_op=tf.summary.merge(summary_ops),
        max_number_of_evaluations=FLAGS.max_number_of_evaluations,
        eval_interval_secs=FLAGS.eval_interval_secs)


if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_dir')
  flags.mark_flag_as_required('eval_logdir')
  flags.mark_flag_as_required('dataset_dir')
  tf.app.run()
