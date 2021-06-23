# Object Detection and Tracking using YOLO and SORT

This project demonstrates real-time object detection and tracking in videos using YOLO (You Only Look Once) for detection and SORT (Simple Online and Realtime Tracking) for tracking.

## Requirements

- Python 3.x
- OpenCV (`pip install opencv-python`)
- CVZone (`pip install cvzone`)
- Ultralytics YOLO (`pip install ultralytics`)

## Setup

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo

## Install dependencies:
```bash
pip install -r requirements.txt
```


## Usage:

- Replace videos_and_images/test2/highway_traffic_flow.mp4 with your video file path for detection.

- Ensure your model weights (yolov8n.pt) are located in model_weights/.

- Adjust the mask region (mask_region.png) and graphic overlays (graphics.png) in the videos_and_images/test2/ directory as per your video setup.

- Run the script:

```bash
python object_detection_tracking.py
```


## View the live video feed:

View the live video feed with object detection and tracking results. Objects detected and tracked will be annotated with bounding boxes and IDs.

