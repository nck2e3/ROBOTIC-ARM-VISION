import multiprocessing
import numpy as np
import cv2
import time
import os
import serial
import sys
import select  # For non-blocking console input on Linux
from pynq.overlays.base import BaseOverlay
from pynq.lib.video import *


# Serial port configuration
SERIAL_PORT = "/dev/ttyACM0"  # Replace with your serial port (e.g., "/dev/ttyUSB0" on Linux, "COM3" on Windows)
BAUD_RATE = 115200            # Baud rate (must match STM32 UART configuration)
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

# Frame Variables
process_frame = 0
faces = None
center_tolerance_px = 50

def process_frames(frame_queue_in, frame_queue_out, cascade_path):
    """Child process to perform face detection on numpy arrays and annotate distance vectors."""
    face_cascade = cv2.CascadeClassifier(cascade_path)
    resized_dims = (800 // 2, 600 // 2)
    scale_x = 800 / resized_dims[0]
    scale_y = 600 / resized_dims[1]
    global process_frame
    global faces
    global center_tolerance_px
    global ser
    while True:
        UART_MSG = "N\n"
        np_frame = frame_queue_in.get()  # Block until we get a numpy array (or None)
        start_time = time.time()  # Start measuring latency

        #Exit sentinel to gracefully exit program if there is no input...        
        if np_frame is None:
            # Received sentinel -> time to exit
            break

        try:
            if process_frame % 2 == 0:
                # Calculate center of the screen based on the current frame dimensions
                height, width = np_frame.shape[:2]
                screen_center = (width // 2, height // 2)

                # Convert to grayscale and resize for face detection
                np_frame_bw = cv2.cvtColor(np_frame, cv2.COLOR_BGR2GRAY)
                resized_frame = cv2.resize(np_frame_bw, resized_dims)

                # Detect faces
                faces = face_cascade.detectMultiScale(resized_frame, 1.4, 2)
                process_frame = 0
            if faces is not None:
                closest_face = None
                min_distance = float('inf')  # Initialize with a very large value
                
                for (x, y, w, h) in faces:
                    # Scale face detection coordinates back to original frame size
                    x = int(x * scale_x)
                    y = int(y * scale_y)
                    w = int(w * scale_x)
                    h = int(h * scale_y)

                    # Compute the face's center (x + w/2, y + h/2)
                    face_center = (int(x + w / 2), int(y + h / 2))

                    # Calculate the distance from the screen center to the face center
                    dx = face_center[0] - screen_center[0]
                    dy = face_center[1] - screen_center[1]
                    distance = (dx**2 + dy**2)**0.5

                    # Update the closest face if this one is closer
                    if distance < min_distance:
                        min_distance = distance
                        closest_face = (x, y, w, h, face_center, dx, dy, distance)

                if closest_face is not None:
                    x, y, w, h, face_center, dx, dy, distance = closest_face
                    color_tuple = (0,0,0)
                    if distance <= center_tolerance_px:
                        UART_MSG = "N\n"  # Message to send
                        color_tuple = (0,255,0)
                    else:
                        UART_MSG = "FLAG\n"
                        color_tuple = (0,0,255)

                    cv2.rectangle(np_frame, (x, y), (x + w, y + h), color_tuple, 2)
                    # Draw rectangle around the closest face
                    # Draw a line from the screen center to the face center
                    cv2.line(
                        np_frame,
                        screen_center,
                        face_center,
                        color_tuple,  # line color (B, G, R)
                        1             # line thickness
                    )

                    # Prepare the text to display
                    distance_text = f"(d={distance:.2f}, dx={dx}, dy={dy})"

                    # Position the text roughly halfway between center and face center
                    text_x = (face_center[0] + screen_center[0]) // 2
                    text_y = (face_center[1] + screen_center[1]) // 2

                    # Draw the text on the frame
                    cv2.putText(
                        np_frame,
                        distance_text,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.33,           # font scale
                        color_tuple,   # text color (B, G, R) - red here
                        1,             # text thickness
                        cv2.LINE_AA
                    )

                
            ser.write(UART_MSG.encode())  # Send as bytes
            
            # Calculate FPS
            latency = time.time() - start_time
            fps = 1.0 / latency if latency > 0 else 0
            fps_text = f"FPS: {fps:.2f}"

            # Annotate FPS on the top-left corner of the frame
            cv2.putText(
                np_frame,
                fps_text,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,            # font scale
                (255, 255, 255), # text color (white)
                1,              # text thickness
                cv2.LINE_AA
            )
            
            process_frame = process_frame + 1
            frame_queue_out.put(np_frame)

        except Exception as e:
            print(f"Error processing frame: {e}")

def main():
    # Environment & overlay
    os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
    base = BaseOverlay("base.bit")
    hdmi_in = base.video.hdmi_in
    hdmi_out = base.video.hdmi_out

    hdmi_in.configure()
    hdmi_out.configure(hdmi_in.mode)
    hdmi_in.cacheable_frames = True
    hdmi_out.cacheable_frames = True
    hdmi_in.start()
    hdmi_out.start()

    # If your design requires tie:
    hdmi_in.tie(hdmi_out)

    # Multiprocessing Queues
    to_process_queue = multiprocessing.Queue(maxsize=1)
    processed_queue = multiprocessing.Queue(maxsize=1)

    # Create child process
    cascade_path = 'data/haarcascade_frontalface_default.xml'
    p = multiprocessing.Process(
        target=process_frames,
        args=(to_process_queue, processed_queue, cascade_path),
        daemon=True
    )
    p.start()

    input_fps = 24
    sleep_interval_seconds = 1 / input_fps
    print("Type anything and press Enter to stop the program...")

    # Main loop to capture and display
    while True:
        # 1) Check if user typed something in the console
        #    select(...) will tell us if sys.stdin has data
        if select.select([sys.stdin], [], [], 0)[0]:
            # Read the line (or at least flush stdin)
            _ = sys.stdin.readline()
            # Break out of the loop to end program
            break

        # 2) Read hardware frame
        pynq_frame = hdmi_in.readframe()

        # 3) Send numpy frame to child process if queue has space
        if not to_process_queue.full():
            to_process_queue.put(pynq_frame)

        # 4) If a processed frame is available, display it
        if not processed_queue.empty():
            processed_np_frame = processed_queue.get()
            out_frame = hdmi_out.newframe()
            out_frame[:] = processed_np_frame  # copy pixels
            hdmi_out.writeframe(out_frame)

        time.sleep(sleep_interval_seconds) #Sleep every frame (input capped @ 30fps for throughput)

    # --- Cleanup ---
    # Send stop signal to child (None signals it's time to exit)
    to_process_queue.put(None)
    p.join(timeout=1)

    hdmi_out.stop()
    hdmi_in.stop()
    del hdmi_in, hdmi_out
    print("Exiting program")


if __name__ == "__main__":
    main()
