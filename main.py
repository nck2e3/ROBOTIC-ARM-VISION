import multiprocessing
import numpy as np
import cv2
import time
import os
import sys
import select  # For non-blocking console input on Linux
from pynq.overlays.base import BaseOverlay
from pynq.lib.video import *

process_frame = 0

def process_frames(frame_queue_in, frame_queue_out, cascade_path):
    """Child process to perform face detection on numpy arrays and annotate distance vectors."""
    face_cascade = cv2.CascadeClassifier(cascade_path)
    resized_dims = (800 // 3, 600 // 3)
    scale_x = 800 / resized_dims[0]
    scale_y = 600 / resized_dims[1]
    global process_frame

    while True:
        np_frame = frame_queue_in.get()  # block until we get a numpy array (or None)
        if np_frame is None:
            # Received sentinel -> time to exit
            break

        start_time = time.time()  # Start measuring latency

        try:
            if process_frame % 4 == 0:
                # Calculate center of the screen based on the current frame dimensions
                height, width = np_frame.shape[:2]
                screen_center = (width // 2, height // 2)

                # Convert to grayscale and resize for face detection
                np_frame_bw = cv2.cvtColor(np_frame, cv2.COLOR_BGR2GRAY)
                resized_frame = cv2.resize(np_frame_bw, resized_dims)

                # Detect faces
                faces = face_cascade.detectMultiScale(resized_frame, 1.3, 5)

                for (x, y, w, h) in faces:
                    # Scale face detection coordinates back to original frame size
                    x = int(x * scale_x)
                    y = int(y * scale_y)
                    w = int(w * scale_x)
                    h = int(h * scale_y)

                    # Draw rectangle around face
                    cv2.rectangle(np_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    # Compute the face's center (x + w/2, y + h/2)
                    face_center = (int(x + w / 2), int(y + h / 2))

                    # Draw a line from the screen center to the face center
                    cv2.line(
                        np_frame,
                        screen_center,
                        face_center,
                        (0, 255, 0),  # line color (B, G, R)
                        1             # line thickness
                    )

                    # Calculate the X and Y distance
                    dx = face_center[0] - screen_center[0]
                    dy = face_center[1] - screen_center[1]

                    # Prepare the text to display
                    distance_text = f"(dx={dx}, dy={dy})"

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
                        (0, 0, 255),   # text color (B, G, R) - red here
                        1,             # text thickness
                        cv2.LINE_AA
                    )
                process_frame = 0

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
    to_process_queue = multiprocessing.Queue(maxsize=2)
    processed_queue = multiprocessing.Queue(maxsize=1)

    # Create child process
    cascade_path = 'data/haarcascade_frontalface_default.xml'
    p = multiprocessing.Process(
        target=process_frames,
        args=(to_process_queue, processed_queue, cascade_path),
        daemon=True
    )
    p.start()

    input_fps = 30
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

        time.sleep(1 / input_fps)

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
