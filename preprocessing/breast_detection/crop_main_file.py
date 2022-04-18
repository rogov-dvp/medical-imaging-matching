#What packages were installed:
# 
# python 3.9.7
# pip 21.2.4
# tensorflow 2.8.0
# opencv
# https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html
# (CPU):
#     - download protobuf version: https://github.com/protocolbuffers/protobuf/releases/tag/v3.19.4
#     - Add to environment path <PROTOBUF_PATH>/bin
#     - run:
#         # From within TensorFlow/models/research/
#         protoc object_detection/protos/*.proto --python_out=.

#         cp object_detection/packages/tf2/setup.py .                 //may need to run this file seperately
#         python -m pip install --use-feature=2020-resolver .

#         python object_detection/builders/model_builder_tf2_test.py
# 

import tensorflow as tf
import os
import math
import cv2
import numpy as np
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.builders import model_builder


WORKSPACE_PATH = 'workspace'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/cropped_images' 
MODEL_PATH = WORKSPACE_PATH+'/models'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mob' 

CUSTOM_MODEL_NAME = 'my_ssd_mob'
# CONFIG_PATH = MODEL_PATH+'/'+CUSTOM_MODEL_NAME+'/pipeline.config'
CONFIG_PATH = '/home/arogov/Documents/src1/preprocessing/breast_detection/workspace/models/my_ssd_mob/pipeline.config'

#config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
# ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
# ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-6')).expect_partial()

# FUNCTIONS
@tf.function
def detect_fn(image):
    print(image)
    image, shapes = detection_model.preprocess(image)
    # print('SHAPES:',shapes)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def crop_breasts(image):
    #category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH + '/label_map.pbtxt')
    image_out = np.copy(image)
    image = np.zeros((image_out.shape[0],image_out.shape[1],3))
    image[:,:,0] = image_out
    image[:,:,1] = image_out
    image[:,:,2] = image_out
    # image = image.astype('float32')
    pre_exp = np.expand_dims(image,0)
    in_tensor = tf.convert_to_tensor(pre_exp, dtype=tf.float32)
    # print(in_tensor.shape)
    detections = detect_fn(in_tensor)
    
    num_detections = int(detections.pop('num_detections'))

    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}

    if detections['detection_scores'][0] < 0.75:
        return image
    
    box = detections['detection_boxes'][0]
    left = math.floor(box[0] * image.shape[1])
    right = math.ceil(box[2] * image.shape[1])
    bot = math.floor(box[1] * image.shape[1])
    top = math.ceil(box[3] * image.shape[1])
    cropped_img = image_out[left:right, bot:top]
        
    return cropped_img


# Set images and run function
# img = cv2.imread("test_images_kaggle/images/2016_BC003122_ CC_L.jpg")  #test_images_kaggle/images
# crop_breasts(np.asarray([img]))  #np.asarray([img,img])