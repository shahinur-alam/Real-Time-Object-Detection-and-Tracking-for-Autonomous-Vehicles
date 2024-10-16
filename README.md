# Real Time Object Detection and Tracking for Autonomous Vehicles

## Overview
This project provides **real-time object detection** using YOLOv8, supporting video input from webcam, uploaded videos, and YouTube links. It uses **Streamlit** for the interface and **OpenCV** for visualization.

## Features
- Real-time object detection with YOLOv8.
- Supports video input from:
  - Webcam
  - Uploaded video files
  - YouTube videos
- Adjustable detection confidence threshold.
- GPU and CPU inference support.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/real-time-object-detection-yolov8.git
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt

3. Clone the repository:
   ```bash
   streamlit run app.py

## Key Functions
- YOLOv8 Model: Loaded using Streamlit’s caching to improve performance.
- VideoProcessor: Processes webcam frames, detects objects, and draws bounding boxes.
- process_uploaded_video(): Processes video files frame by frame for detection.
- process_youtube_video(): Downloads and processes YouTube videos for detection.

  ## Usage
1. Webcam: Select "Webcam" and allow access to your webcam.
2. Upload Video: Upload a video file (.mp4, .mov, .avi).
3. YouTube Video: Enter a YouTube URL to process the video.

  ## Troubleshooting
- Ensure correct installation of PyTorch, YOLOv8, and CUDA (if using GPU).
- Start the app using streamlit run app.py.
