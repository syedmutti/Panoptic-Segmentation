#!/usr/bin/env python
# coding: utf-8

import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib
import time
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import glob 
import cv2

import tensorflow as tf



class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  OUTPUT_TENSOR_INSTANCE_NAME = 'InstanceCenterPredictions:0'
  OUTPUT_TENSOR_OFFSET_NAME = 'InstanceOffsetPredictions:0'
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, GRAPH_PB_PATH):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()
    graph_def = None
    
    with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())


    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    """
    t1 = time.time()
    batch_seg_map, batch_instance_center, batch_instance_offset = self.sess.run(
        [self.OUTPUT_TENSOR_NAME, self.OUTPUT_TENSOR_INSTANCE_NAME, self.OUTPUT_TENSOR_OFFSET_NAME],
        feed_dict={self.INPUT_TENSOR_NAME: image})
    t2 = time.time()
    print(t2-t1)


    seg_map = batch_seg_map[0]
    batch_instance_center = batch_instance_center[0]
    batch_instance_offset = batch_instance_offset[0]

    return seg_map, batch_instance_center, batch_instance_offset





from tensorflow.python.platform import gfile
MODEL = DeepLabModel("tmp_graph/frozen_inference_graph.pb")
print('model loaded successfully!')



image_path = glob.glob('/mrtstorage/users/rehman/datasets/cityscapes/leftImg8bit/train/aachen/*.png')

image1 = Image.open(image_path[1])  #65

image = np.expand_dims(image1, 0)


seg_map, center_pred, pred_offset = MODEL.run(image)



# Post-Processing to visualize Outputs 
print(pred_offset.shape)

offset = pred_offset
red, green = np.dsplit(offset, 2)
max_red = np.amax(red)
red = np.multiply(np.divide(red, max_red), 255.0)
max_green = np.amax(green)
green = np.multiply(np.divide(green, max_green), 255.0)
blue = np.zeros_like(red)
offset = np.concatenate((red, green, blue), 2)

print(offset.shape)

original_image = Image.fromarray(seg_map* 255//19)
max_center = np.amax(center_pred)
instance_center = np.multiply(np.divide(center_pred, max_center), 255.0)



def nms_centerpoint_stride(image, kernel_size=16):
    test =np.array(image)
    
    center_dict = {}
    for h in range(test.shape[0] // kernel_size):
        for w in range(test.shape[1] // kernel_size-1):

            local_region = test[h*kernel_size: (h*kernel_size) +kernel_size, w*kernel_size :(w*kernel_size) + kernel_size]
            local_max = np.amax(local_region)
            if local_max > 25.5:
                idx = np.where(test == local_max)
            # Dictionary
                center_dict[local_max] = idx
    return center_dict


# Max Pooling Based Non-maxima supression

max_center = np.amax(center_pred)
nms_pooling_filer_size = 32
instance_center = np.multiply(np.divide(center_pred, max_center), 255.0)
center_dict = nms_centerpoint_stride(instance_center, nms_pooling_filer_size)
print(len(center_dict))

new_array = np.zeros(np.array(instance_center).shape)
center_points = []
for index, key in enumerate(sorted(center_dict.keys(), reverse=True)):
    if index < 200:
        center_points.append(center_dict[key])
instance_resoltion = 16
filtered_centers = center_points
for center in center_points:
    for i, center_x in enumerate(center_points):
        if center_x != center:
            diff_center_h = np.square(np.subtract(center[0], center_x[0]))
            diff_center_w = np.square(np.subtract(center[1], center_x[1]))
            diff_dist = np.sqrt(diff_center_h + diff_center_w)
            if diff_dist < instance_resoltion:
                filtered_centers.pop(i)

print('number of instances_found : {}'.format(len(filtered_centers)))

for center in filtered_centers:
    new_array[center]= 255
Image.fromarray(new_array.astype('uint8'))


# Mask for Instances from semantic segmentation

semantic_seg_array = seg_map
unique_ids = np.unique(semantic_seg_array)
print('Instances : {}'.format(unique_ids))
instances_in_image = [unique_id for unique_id in unique_ids if(unique_id > 10 and unique_id <= 18)]
panoptic_mask = np.zeros(np.array(seg_map).shape)

for i in range(len(instances_in_image)):
    
    local_mask = semantic_seg_array == instances_in_image[i]
    panoptic_mask += local_mask
    
panoptic_mask = panoptic_mask.astype('bool')

Image.fromarray(panoptic_mask)


# Generate Instance Segmentation 
def assign_id(center, ij, w_vectors, h_vectors):
    h, w = ij 
    index_regression = []
    for i in range(len(center)):

        if (center[i][1] > w):
            index_plus_offset_w = np.add(w, w_vectors[h][w])
        elif (center[i][1] <= w):
            index_plus_offset_w = np.subtract(w, w_vectors[h][w])
        if (center[i][0] > h):
            index_plus_offset_h = np.add(h, h_vectors[h][w])
        elif (center[i][0] <= h):
            index_plus_offset_h = np.subtract(h, h_vectors[h][w])
            
        index_plus_offset = [index_plus_offset_h, index_plus_offset_w]
        diff_center_h = np.square(np.subtract(center[i][0], index_plus_offset[0]))
        diff_center_w = np.square(np.subtract(center[i][1], index_plus_offset[1]))
        diff_dist = np.square(np.sqrt(np.add(diff_center_h, diff_center_w)))
        index_regression.append(diff_dist)

    x = np.argmin(index_regression)
    return x+1


# Data Preperation
semantic_seg = Image.fromarray(seg_map)
panoptic_segmentation = Image.new('L', semantic_seg.size)
panoptic_segmentation = np.array(panoptic_segmentation)
red, green = np.dsplit(pred_offset, 2)
x_vectors = red * 255
y_vectors = green * 255


for center in filtered_centers:
    panoptic_mask[center] = False


instance_pixels = np.where(panoptic_mask)
instance_pixels = np.array(instance_pixels).transpose()


start = time.time()
print(len(instance_pixels))
for i in range(len(instance_pixels)):
    h, w = instance_pixels[i]   
    panoptic_segmentation[h][w] = assign_id(center_points, [h,w], x_vectors, y_vectors)

end_time = time.time()

panoptic_segmentation_scaled = panoptic_segmentation* (255//len(np.unique(panoptic_segmentation)))
Image.fromarray(panoptic_segmentation_scaled)
print(end_time - start)


instance_seg_img = Image.fromarray(panoptic_segmentation_scaled)

inst_color = cv2.applyColorMap(panoptic_segmentation_scaled, cv2.COLORMAP_JET)
inst_color = Image.fromarray(inst_color)

Image.blend(image1, inst_color, 0.4)


# Majority Voting for ID assignment

instances = np.unique(panoptic_segmentation)
instances = np.delete(instances, 0)
print(instances)
for index, i in enumerate(instances):
    local_instance_mask = panoptic_segmentation == i
    local_instance_seg = seg_map * local_instance_mask
    
    unique_elements, counts_elements = np.unique(local_instance_seg, return_counts=True)
    zero = np.argmax(counts_elements)
    if unique_elements[zero] == 0:
        unique_elements = np.delete(unique_elements, zero)
        counts_elements = np.delete(counts_elements, zero)


    majority_id  = unique_elements[np.argmax(counts_elements)]

    print('{}: Id of instance is {}'.format(index+1, majority_id))
    
# Add colormap to segmentation visualization 

import get_dataset_colormap
colormap_type = 'cityscapes'

# Add boundry to Image 
instance_boundry = np.zeros_like(seg_map)

instances = np.unique(panoptic_segmentation)
instances = np.delete(instances, 0)
print(instances)
    
for index, i in enumerate(instances):
    local_instance_mask = panoptic_segmentation == i
    kernel = np.ones((5,5),np.uint8)
   
    dilation2 = cv2.dilate(local_instance_mask.astype('uint8'), kernel, iterations = 1)
    erosion2 = cv2.erode(dilation2, kernel, iterations = 1)
    boundry = (dilation2 - erosion2) * 255
    instance_boundry += boundry

#Image.fromarray(instance_boundry.astype('uint8')).show()    

colored_label = get_dataset_colormap.label_to_color_image(
         np.array(seg_map), colormap_type)

colored_label = Image.fromarray(colored_label.astype(dtype=np.uint8))

instance_boundry = np.dstack((instance_boundry, instance_boundry, instance_boundry))
instance_boundry_img = Image.fromarray(instance_boundry.astype('uint8'))


panoptic_seg = Image.blend(colored_label, instance_boundry_img, 0.5 )


panoptic_seg = Image.blend(panoptic_seg, image1, 0.5 )



Image.fromarray(panoptic_seg).show()






