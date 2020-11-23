import tensorflow as tf
from yolo.utils import load_class_names, output_boxes, draw_outputs, resize_image
import cv2
import os
import numpy as np
from yolo.yolov3 import YOLOv3Net


# physical_device = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_device) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_device[0], True)

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


model_size = (416, 416, 3)
number_classes = 80
class_name = 'yolo/data/coco.names'
max_output_size = 40
max_output_size_per_class = 20
iou_threshold = 0.5
confidence_threshold = 0.5

cfgfile = 'yolo/cfg/yolov3.cfg'
weightfile = 'yolo/weights/yolov3_weights.tf'
# img_path = "img/kitchen.jpg"


def detect_image(img_path):

    model = YOLOv3Net(cfgfile, model_size, number_classes)
    model.load_weights(weightfile)

    class_names = load_class_names(class_name)

    # read image
    # image = cv2.imread(img_path)
    # convert image to ndarray image
    image = np.array(img_path)

    """
     This operation is useful if you want to add a batch dimension to a single element. 
     For example, if you have a single image of shape [height, width, channels], you can make it a 
     batch of 1 image with expand_dims(image, 0), which will make the shape [1, height, width, channels].
    """
    image = tf.expand_dims(image, 0)
    resize_frame = resize_image(image, (model_size[0], model_size[1]))
    pred = model.predict(resize_frame)

    boxes, scores, classes, nums = output_boxes(
        pred, model_size,
        max_output_size = max_output_size,
        max_output_size_per_class = max_output_size_per_class,
        iou_threshold = iou_threshold,
        confidence_threshold=confidence_threshold)

    # Convert back to ndarray image. The input array, but with all or a subset of the dimensions of length 1 removed.
    image = np.squeeze(image)

    img, obj_name = draw_outputs(image, boxes, scores, classes, nums, class_names)
    labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
              "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "chair", "refrigerator"]
    object_with_label = {}
    for key, val in obj_name.items():
        if key in labels:
            object_with_label[key] = val
            print('{} {:.4f}'.format(key, val))
    return object_with_label
