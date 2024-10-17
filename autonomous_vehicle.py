import os
import sys
import cv2
import numpy as np
import streamlit as st
import av
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import yt_dlp

# Set environment variable to handle OpenMP error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Ensure the correct Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    import torch
    from ultralytics import YOLO
except ImportError as e:
    st.error(f"Error importing required modules: {e}")
    st.error("Please ensure you have PyTorch and Ultralytics installed correctly.")
    st.stop()


# Load YOLOv8 model
@st.cache_resource
def load_model():
    try:
        return YOLO('yolov8n.pt')
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None


model = load_model()

if model is None:
    st.error("Failed to load the YOLO model. Please check your installation and try again.")
    st.stop()


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.confidence_threshold = 0.5

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Perform inference
        results = model(img, device='cuda' if torch.cuda.is_available() else 'cpu')

        # Draw bounding boxes and labels
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if box.conf.item() > self.confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{model.names[int(box.cls)]} {box.conf.item():.2f}"
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def process_uploaded_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    vf = cv2.VideoCapture(tfile.name)
    stframe = st.empty()
    while vf.isOpened():
        ret, frame = vf.read()
        if not ret:
            break
        results = model(frame)
        res_plotted = results[0].plot()
        stframe.image(res_plotted, channels="BGR")
    vf.release()
    os.unlink(tfile.name)


def process_youtube_video(youtube_url):
    try:
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': 'temp_video.%(ext)s'
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        cap = cv2.VideoCapture('temp_video.mp4')
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            res_plotted = results[0].plot()
            stframe.image(res_plotted, channels="BGR")
        cap.release()
        os.remove('temp_video.mp4')
    except Exception as e:
        st.error(f"Error processing YouTube video: {e}")


def main():
    st.title("Real-Time Object Detection with YOLOv8")

    st.write("System Information:")
    st.write(f"Python version: {sys.version}")
    st.write(f"PyTorch version: {torch.__version__}")
    st.write(f"OpenCV version: {cv2.__version__}")
    st.write(f"CUDA available: {torch.cuda.is_available()}")

    option = st.selectbox("Select Input Source", ["Webcam", "Upload Video", "YouTube Video"])

    if option == "Webcam":
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

        try:
            webrtc_ctx = webrtc_streamer(
                key="object-detection",
                video_processor_factory=VideoProcessor,
                rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                media_stream_constraints={"video": True, "audio": False},
            )

            if webrtc_ctx.video_processor:
                webrtc_ctx.video_processor.confidence_threshold = confidence_threshold
        except Exception as e:
            st.error(f"Error setting up webcam stream: {e}")
            st.info("If you're running this locally, make sure to use 'streamlit run' command.")

    elif option == "Upload Video":
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
        if uploaded_file is not None:
            process_uploaded_video(uploaded_file)

    elif option == "YouTube Video":
        youtube_url = st.text_input("Enter YouTube URL")
        if youtube_url:
            process_youtube_video(youtube_url)


if __name__ == "__main__":
    main()