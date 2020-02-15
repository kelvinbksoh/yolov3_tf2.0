from model import YoloV3, yolo_anchors, yolo_anchor_masks
from utils import initialize_yolov3_weights

"""
This script should only be run whenever you want to convert any new darknet weights
into tensorflow weights format.

1. Initialize and build the YoloV3 model from the model class
2. Perform the weight conversions of darknet weights to tensorflow weights
3. Run detect.py
"""

darknet_weights = "./lock_only_top_view_default_1900.weights"
output = "./yolov3_.tf"


yolo = YoloV3(yolo_anchors, yolo_anchor_masks, 1)
##yolo.build((416,416,3))
#_ = yolo(tf.ones([1,416,416,3]))
initialize_yolov3_weights(yolo, darknet_weights, output)