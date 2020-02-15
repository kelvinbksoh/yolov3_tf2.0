# YoloV3_tf2.0

YoloV3 implemented with tensorflow 2.0 using keras subclassing API. Model subclassing gives us the flexibility to customize any kind of NN architecture not provided by the default Sequential API.

### convert.py

This script convert any darknet weights into tensorflow .tf weights format which are needed for tensorflow.
This should only run once whenever you want to convert any new darknet weights into .tf weights.

```Bash
python convert.py
```

### detect.py
YoloV3 algorithm to perform object detections.

```Bash
python detect.py
```