import cv2
import tensorflow as tf
from model import YoloV3, anchors, anchor_masks
import matplotlib.pyplot as plt
from utils import resize_images, draw_outputs

image = './lock.jpg'
weights_path = "./yolov3_.tf"

def initialize_yolov3_model(model, weights_path):
    """Initialize the Yolov3 model with the loaded weights

    Args:
        model: YoloV3 model
        weights_file: Tensorflow weights file 
        
    Returns:
        yolo: Our yolov3 model with the loaded weights
    """
    #yolo = YoloV3(classes=1)
    yolo = model
    _ = yolo(tf.ones([1,416,416,3])) # Need to call this first to instantiate the model to some random weights
    #yolo.model((416, 416, 3)) # Showing the model summary
    yolo.load_weights(weights_path) # Instantiate our model with our darknet weights    
    return yolo


def detect(model, image, image_size):
    """Perform detection on image using the YoloV3 model

    Args:
        model: YoloV3 model
        image: Image to be detected
        image_size: Size of the image to be resized to

    """
    img_raw = tf.image.decode_image(
                open(image, 'rb').read(), channels=3)
    
    img = tf.expand_dims(img_raw, 0)
    img = resize_images(img, image_size)
    boxes, scores, classes, nums = yolo(img) # Calling our yolo model on the image
    
    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR) # openCV BGR format
    img = draw_outputs(img, (boxes, scores, classes, nums), ['lock'])
    
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cv2.imwrite("lock_final_.jpg", img)

if __name__ == '__main__':
    # Instantiate the model
    yolo = initialize_yolov3_model(YoloV3(anchors, anchor_masks, 1), weights_path)

    # Perform the detection on the image of size 416x416
    detect(yolo, image, 416)