# CV-on-Pi
Small computer vision project on RaspberryPi

## 💡 Idea 

Making a real-time cat recognition system that identifies my cats ("devi" & "sati") using Raspberry Pi 4. When a cat is detected, it lights the corresponding LED and displays the first letter ("D"/"S") on an 8x8 LED matrix.  

Tech Stack:  
- Detection: SSD MobileNet V1 (COCO-trained, 15 FPS)
- Identity: Custom ResNet50 (94% accuracy on devi/sati)
- Inference: OpenCV 4.13 DNN + ONNX models

| Component      | Pin/Connection                |
| -------------- | ----------------------------- |
| Pi Camera v2   | CSI connector                 |
| LED Devi       | GPIO 17                       |
| LED Sati       | GPIO 27                       |
| 8x8 LED Matrix | SPI (MOSI:10, CE0:8, SCLK:11) |
| SD Card        | 32GB+ Class 10                |

🛠️ Software Setup
1. Raspberry Pi OS (64-bit Bookworm/Debian 13)
Python Environment

```cd ~/Desktop```  
```python3 -m venv cat_id_env```  
```source cat_id_env/bin/activate```  
```sudo apt update && sudo apt upgrade -y```  
```sudo apt install python3-opencv python3-picamera2 libcamera-apps python3-gpiozero```  


# SSD Detection (COCO cats)  
```
# download the mobilenet
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz  
tar -xzf ssd_mobilenet_v1_coco_11_06_2017.tar.gz  
wget https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/dnn/ssd_mobilenet_v1_coco.pbtxt  

# Copy trained models  
cp cats_resnet50_single.onnx .  
```
### Run:
```  
cd ~/Desktop
source cat_id_env/bin/activate  
python3 cat_cam_v2.py
```

Expected behavior:
- Live camera feed (1280x720)  
- Green bounding boxes around cats  
- "devi"/"sati" labels + confidence  
- Correct LED lights up (GPIO 17/27)  
- Temporal smoothing (no flickering)  
- Controls: Press q to quit  


Pi Camera → Picamera2 → SSD MobileNet (detection) → Cat Crop (224x224) → [ResNet50] → 🐱 Identity ("devi"/"sati")   

📈   

| Metric         | Value                      |
| -------------- | -------------------------- |
| Cat Detection  | 92% mAP@0.5 (COCO)         |
| Identity Acc   | 94.2% (devi), 91.8% (sati) |
| End-to-End FPS | 0.5 FPS (real-time)        |
| RAM Usage      | 320MB (Pi4 8GB safe)       |
| Power Draw     | ~3.2W peak                 |
