import asyncio
from pynq.overlays.base import BaseOverlay
from pynq.lib.video import *
import cv2
import threading
import os
import numpy as np
import time

class FrameBuffer:
    """Abstracts frame buffer operations."""
    def __init__(self, maxsize):
        from queue import Queue
        self.buffer = Queue(maxsize=maxsize)

    def add_frame(self, frame):
        """Add a frame to the buffer if there is space."""
        if not self.buffer.full():
            self.buffer.put(frame)

    def get_frame(self):
        """Retrieve a frame from the buffer if available."""
        if not self.buffer.empty():
            return self.buffer.get()
        return None

    def is_empty(self):
        """Check if the buffer is empty."""
        return self.buffer.empty()

    def is_full(self):
        """Check if the buffer is full."""
        return self.buffer.full()

def capture_frames(hdmi_in, frame_buffer):
    """Thread to capture frames and add to buffer."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    input_fps = 24

    while True:
        # Capture a frame from HDMI input
        frame = hdmi_in.readframe()

        # Add frame to buffer
        frame_buffer.add_frame(frame)

        time.sleep(1 / input_fps)

def process_frames(input_buffer, output_buffer, face_cascade):
    """Optimized frame processing for real-time face detection."""
    resized_dims = (200, 150)  # Half of 800x600
    scale_x = 800 / resized_dims[0]
    scale_y = 600 / resized_dims[1]

    while True:
        frame = input_buffer.get_frame()
        if frame is not None:
            try:
                # Convert to grayscale
                # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Resize the grayscale frame
                resized_frame = cv2.resize(frame, resized_dims)

                # Detect faces in the resized frame
                faces = face_cascade.detectMultiScale(
                    resized_frame,
                    scaleFactor=1.3,
                    minNeighbors=1,
                    minSize=(5,5)
                )

                # Scale detected faces back to the original frame size
                for (x, y, w, h) in faces:
                    x = int(x * scale_x)
                    y = int(y * scale_y)
                    w = int(w * scale_x)
                    h = int(h * scale_y)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

                # Add processed frame to the output buffer
                output_buffer.add_frame(frame)

            except Exception as e:
                print(f"Error processing frame: {e}")


def display_frames(processed_buffer, hdmi_out):
    """Thread to display processed frames."""

    while True:
        frame = processed_buffer.get_frame()
        if frame is not None:
            hdmi_out.writeframe(frame)

def main():
    # Configure environment
    os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

    # Load the overlay
    base = BaseOverlay("base.bit")
    hdmi_in = base.video.hdmi_in
    hdmi_out = base.video.hdmi_out

    # Configure HDMI interfaces
    hdmi_in.configure(PIXEL_GRAY)
    hdmi_out.configure(hdmi_in.mode)
    hdmi_in.cacheable_frames = True
    hdmi_out.cacheable_frames = True
    hdmi_in.start()
    hdmi_out.start()

    # Tie input to output...
    hdmi_in.tie(hdmi_out)

    # Initialize frame buffers
    capture_buffer = FrameBuffer(maxsize=1)
    processed_buffer = FrameBuffer(maxsize=2)

    # Face detection cascade
    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

    # Create and start threads
    capture_thread = threading.Thread(target=capture_frames, args=(hdmi_in, capture_buffer), daemon=True)
    processing_thread = threading.Thread(target=process_frames, args=(capture_buffer, processed_buffer, face_cascade), daemon=True)
    display_thread = threading.Thread(target=display_frames, args=(processed_buffer, hdmi_out), daemon=True)

    capture_thread.start()
    processing_thread.start()
    display_thread.start()

    # Wait for user input to stop the program
    input("Press Enter to stop the program...")

    # Clean up
    hdmi_out.stop()
    hdmi_in.stop()
    del hdmi_in, hdmi_out
    print("Exiting program")

if __name__ == "__main__":
    main()
