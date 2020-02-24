# YoloV3_tf2.0

YoloV3 implemented with tensorflow 2.0 in an object-oriented approach using keras subclassing API. Model subclassing gives us the flexibility to customize exotic architectures not provided by the default Sequential/Functional API.

### convert.py
This script convert any darknet weights into tensorflow .tf weights format which are needed for tensorflow.
This should only run once whenever you want to convert any new darknet weights into .tf weights.

```Bash
python convert.py
```
### model.py
This script defines the YoloV3 architecture as an object-oriented approach, it gives us full flexibility to deep dive into the layers, sublayers and models.

### detect.py
YoloV3 algorithm to perform object detections.

```Bash
python detect.py
```

TODO..
Writing custom loss function using GIOU and IOU for the bounding boxes.
