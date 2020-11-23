import tensorflow as tf
from yolo.utils import load_class_names, output_boxes, draw_outputs, resize_image
from yolo.yolov3 import YOLOv3Net
import os
import cv2
import time

# physical_device = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_device) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_device[0], True)

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


model_size = (416, 416, 3)
number_classes = 80
class_name = 'data/coco.names'
max_output_size = 100
max_output_size_per_class = 20
iou_threshold = 0.5
confidence_threshold = 0.7

cfgfile = 'cfg/yolov3.cfg'
weightfile = 'weights/yolov3_weights.tf'


def main():

    model = YOLOv3Net(cfgfile, model_size, number_classes)
    model.load_weights(weightfile)
    class_names = load_class_names(class_name)

    win_name = 'Yolov3 detection'
    cv2.namedWindow(win_name)

    #specify the video input

    cap = cv2.VideoCapture('rtsp://admin:a123456789@218.151.33.75:554/Streaming/channels/301')
    #cap = cv2.VideoCapture('rtsp://admin:rex6885!@sel312.iptime.org:20004/MOBILE/media.smp')
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                  cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    try:
        while True:
            start = time.time()
            ret, frame = cap.read()

            if not ret:
                break

            resized_frame = tf.expand_dims(frame, 0)
            resized_frame = resize_image(resized_frame, (model_size[0], model_size[1]))

            pred = model.predict(resized_frame)

            boxes, scores, classes, nums = output_boxes(
                pred, model_size,
                max_output_size=max_output_size,
                max_output_size_per_class=max_output_size_per_class,
                iou_threshold=iou_threshold,
                confidence_threshold=confidence_threshold)

            img, obj_name = draw_outputs(frame, boxes, scores, classes, nums, class_names)

            labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
                      "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]
            for key, val in obj_name.items():
                if key in labels:
                    print('{} {:.4f}'.format(key, val))

            cv2.imshow(win_name, img)

            stop = time.time()

            seconds = stop - start

            # calculate frame per second
            fpt = 1 / seconds
            print("Estimated frames per second : {0}".format(fpt))

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        cap.release()
        print('Detections have been performed successfully.')


if __name__ == '__main__':
    main()