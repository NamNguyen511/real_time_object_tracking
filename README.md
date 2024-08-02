# Real-Time Object Tracking

This project implements various basic techniques for real-time object tracking using OpenCV in Python. For further implementations, please follow OpenCV documentaion.

- Motion Detection
- Optical Flow 
- Background Subtraction
- Object Tracking with MeanShift and CamShift

## Setup
1. Clone the repository:
    ```sh
   git clone https://github.com/yourusername/real_time_object_tracking.git
   cd real_time_object_tracking

2. Install the required packages:
    ```sh
   pip install -r requirements.txt
   
## Usage
Run the `motion_detection.py` script to start the real_time object tracking application. The same with other files.
   ```sh
   python motion_detection.py
```
![Screenshot_2](https://github.com/user-attachments/assets/ca128c5b-c987-459a-bb71-3640f943576c)

## Features
- Motion Detection: Detects and highlights moving objects in the video feed
- Optical Flow: Computes and visualizes the motion vectors in the video
- Background Subtraction: Separates moving objects from the background using MOG2 and KNN methods.
- Object Tracking: Tracks the selected object using MeanShift and CamShift

## Contributing
Feel free to submit issues or pull request if you have any improvements or bug fixes.
