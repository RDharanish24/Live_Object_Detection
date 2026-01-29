# Real-Time Object Detection using MobileNet SSD (OpenCV)

This project performs real-time object detection using a webcam with the MobileNet SSD (Single Shot Detector) model and OpenCV’s DNN module. It detects common objects such as people, cars, bottles, animals, etc., and draws bounding boxes with confidence scores.

---

## 🚀 Features

Real-time object detection using webcam

Pre-trained MobileNet SSD (Caffe model)

Lightweight and fast (runs on CPU)

Bounding boxes with class labels & confidence

Option to save detected output as a video (output.avi)

##
📁 Project Structure
```
object-detection/
│
├── models/
│   ├── MobileNetSSD_deploy.prototxt
│   └── MobileNetSSD_deploy.caffemodel
│
├── output.avi              # Generated after running the program
├── main.py                 # Python script
├── requirements.txt
└── README.md
```
## 🛠️ Requirements

Install dependencies using:
```
pip install -r requirements.txt
```
requirements.txt
```
numpy==1.26.4
opencv-python==4.9.0.80
```

## 🔽 Download Pre-trained Model Files (Important)

Before running the project, you must download the MobileNet SSD model files and place them inside the models/ folder.

Required files:
```
MobileNetSSD_deploy.prototxt

MobileNetSSD_deploy.caffemodel
```


Folder structure after download:
```
models/
├── MobileNetSSD_deploy.prototxt
└── MobileNetSSD_deploy.caffemodel
```

### ⚠️ Note:
If these files are missing or paths are incorrect, OpenCV will throw an error while loading the network.

## ▶️ How to Run

Clone the repository or download the project

Ensure your webcam is connected

Run the script:
```

python main.py

```
Press q to exit the program

## 📷 Supported Object Classes
```
background, aeroplane, bicycle, bird, boat,
bottle, bus, car, cat, chair, cow, diningtable,
dog, horse, motorbike, person, pottedplant,
sheep, sofa, train, tvmonitor
```

##
⚙️ How It Works (High Level)

Capture frames from webcam

Convert frame to a blob

Pass blob through MobileNet SSD

Filter detections by confidence threshold

Draw bounding boxes and labels

Display and optionally save output video

##
❗ Common Issues

Camera not opening → Check webcam permissions or index (VideoCapture(0))

Model not loading → Verify correct paths to .prototxt and .caffemodel

Low FPS → Reduce resolution or confidence threshold

##
📈 Future Improvements

Switch to YOLO / ONNX

GPU (CUDA) acceleration

FPS counter

Image / video file detection

Streamlit or Flask UI