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
CONFIG_PATH = MODEL_PATH+'/'+CUSTOM_MODEL_NAME+'/pipeline.config'




config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-6')).expect_partial()

# FUNCTIONS
@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def crop_breasts(images):
    category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH + '/label_map.pbtxt')

    images_out = []

    for i in range(images.shape[0]):
        in_tensor = tf.convert_to_tensor(np.expand_dims(images[i],0), dtype=tf.float32)
        detections = detect_fn(in_tensor)

        num_detections = int(detections.pop('num_detections'))

        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}

        if detections['detection_scores'][0] < 0.75:
            print('Score < 0.75 for i=' + str(i))
        box = detections['detection_boxes'][0]

        image_np_crop = images[i].copy()
        left = math.floor(box[0] * images.shape[1])
        right = math.ceil(box[2] * images.shape[1])
        bot = math.floor(box[1] * images.shape[1])
        top = math.ceil(box[3] * images.shape[1])
        cropped_img = image_np_crop[left:right, bot:top]
        name = './Image_' + str(i) + '_crop.jpeg'       #
        cv2.imwrite(IMAGE_PATH +'/{}'.format(name) , cropped_img)

    
    return np.asarray(images_out)

# This function exists for reading cv2
def set_image(file_location):
    return cv2.imread(file_location)


#Set images and run function
img = set_image("../../test_images_kaggle/images/2017_BC015902_ CC_L.jpg")  #TODO: Image needs to be automatically inserted
crop_breasts(np.asarray([img]))  #np.asarray([img,img])