import numpy as np
import tensorflow as tf
import cv2

def resize_images(image, size):
    images = tf.image.resize(image, (size, size)) # Assuming width and heights are the same size
    images = images / 255.0 # Normalizes the image to range 0-1
    return images

def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (0, 255, 0), 3)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 0), 2)
    return img

def convert_darknet_weights(yolov3_list, weights_file):
    """Convert the darknet into tensorflow weights format

    Args:
        yolov3_list: A list of all Yolov3 architecture layers
        weights_file: A darknet .weights file 

    Returns:
    """
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
    
    for i, layer in enumerate(yolov3_list):
        if not layer.name.startswith('conv2d'):
            continue
        batch_norm = None
        if i < len(yolov3_list) - 1:
            if yolov3_list[i+1].name.startswith('batch_norm'):
                batch_norm = yolov3_list[i+1]
        filters = layer.filters
        size = layer.kernel_size[0]
        in_dim = layer.input_shape[-1]
        
        if batch_norm is None:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
        else:
            bn_weights = np.fromfile(
                wf, dtype=np.float32, count=4*filters)
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            
        conv_shape = (filters, in_dim, size, size)
        conv_weights = np.fromfile(
            wf, dtype=np.float32, count=np.product(conv_shape))
        conv_weights = conv_weights.reshape(
            conv_shape).transpose([2, 3, 1, 0])
            
        if batch_norm is None:
            layer.set_weights([conv_weights, conv_bias])
        else:
            layer.set_weights([conv_weights])
            batch_norm.set_weights(bn_weights)
    
#    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def get_layers_from_submodel(model, sub_model_name, with_bn=True):
    """Retrieve all layers from our keras.model sublayer

    Args:
        model: Yolo model
        sub_model_name: The name of the keras.model sub_model
        with_bn: If true then tehre is batch_norm after every convolution layer
        
    Returns:
        A list of sub_model layers        
    """
    layers_from_submodel = []
    if with_bn:
        for conv_with_bn in model.get_layer(sub_model_name).layers:
            layers_from_submodel.append(conv_with_bn.conv)
            layers_from_submodel.append(conv_with_bn.bn_layer)
    else:
        for i, conv_with_bn in enumerate(model.get_layer(sub_model_name).layers):
            if (i == 0):
                layers_from_submodel.append(conv_with_bn.conv)
                layers_from_submodel.append(conv_with_bn.bn_layer)
            else:
                layers_from_submodel.append(conv_with_bn.conv)
    return layers_from_submodel

def initialize_yolov3_weights(model, darknet_weights, output):
    
    """Initialize and assign darknet weights to YoloV3 weights, then save to a .tf file

    Args:
        model: A list of all Yolov3 architecture layers
        darknet_weights: A darknet .weights to be converted 
        output: output of the converted .tf weights file
    """
#    yolo = YoloV3(yolo_anchors, yolo_anchor_masks, 1)
    yolo = model
    #yolo.build((416,416,3))
    _ = yolo(tf.ones([1,416,416,3]))
    yolo.model((416, 416, 3)) # instantiate the model


    darknet_list = []
    darknet_layers = yolo.get_layer('darknet53').layers
    for i, conv_with_bn in enumerate(darknet_layers):
        if (conv_with_bn.name.startswith('conv')):
            print(i, conv_with_bn.name)
            conv = darknet_layers[i].conv 
            bn = darknet_layers[i].bn_layer
            
            darknet_list.append(conv)
            darknet_list.append(bn)
        if (conv_with_bn.name.startswith('residual')):
            print(i, conv_with_bn.name)
            for residual_blk in darknet_layers[i].layers:
                    
                    conv_1 = residual_blk.conv_1.conv
                    bn_1 = residual_blk.conv_1.bn_layer
            
                    conv_2 = residual_blk.conv_2.conv
                    bn_2 = residual_blk.conv_2.bn_layer
            
                    darknet_list.append(conv_1)
                    darknet_list.append(bn_1)
                    darknet_list.append(conv_2)
                    darknet_list.append(bn_2)
                    
    
    """
    Start
        Below are the layers extraction process for the the Yolo layers
    """
    yolo_conv_blk_large = get_layers_from_submodel(yolo, 'yolo_conv_blk_large')
    yolo_detections_large = get_layers_from_submodel(yolo, 'yolo_detections_large', False)
    
    yolo_conv_blk_medium = get_layers_from_submodel(yolo, 'yolo_conv_blk_medium')
    yolo_detections_medium = get_layers_from_submodel(yolo, 'yolo_detections_medium', False)
    
    yolo_conv_blk_small = get_layers_from_submodel(yolo, 'yolo_conv_blk_small')
    yolo_detections_small = get_layers_from_submodel(yolo, 'yolo_detections_small', False)
    
    tensors_list = [yolo_conv_blk_large, yolo_detections_large, yolo_conv_blk_medium, \
                    yolo_detections_medium, yolo_conv_blk_small, yolo_detections_small]
    
    flatten_tensors_list = [item for sublist in tensors_list for item in sublist]
    """
    End
        Yolo layers
    """

    yolov3_list = darknet_list + flatten_tensors_list # Concatenate the extracted darknet layers and the Yolo layers    
    convert_darknet_weights(yolov3_list, darknet_weights) # Perform the conversion of darknet weights to .tf checkpoints
    yolo.save_weights(output) # Save the converted darknet weights to a .tf file