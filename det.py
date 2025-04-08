import cv2
import time
from ultralytics import YOLO
import numpy as np
import psutil
import os
import threading
from queue import Queue

class WebcamVideoStream:
    """
    Threaded webcam capture for improved performance.
    """
    def __init__(self, src=0, width=640, height=480):
        # Initialize camera
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Try to disable camera auto features if supported
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
        self.stream.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
        
        # Set camera to highest frame rate available
        self.stream.set(cv2.CAP_PROP_FPS, 60)
        
        # Read first frame to initialize
        (self.grabbed, self.frame) = self.stream.read()
        
        # Variable to control thread operation
        self.stopped = False
        
        # Performance metrics
        self.capture_times = []
        self.last_time = time.time()
        
    def start(self):
        # Start the thread
        threading.Thread(target=self.update, args=()).start()
        return self
        
    def update(self):
        # Run indefinitely until thread is stopped
        while not self.stopped:
            # Record capture time
            start_time = time.time()
            
            # Grab frame (separate to minimize time between frames)
            grabbed = self.stream.grab()
            
            if grabbed:
                # Decode the grabbed frame
                _, self.frame = self.stream.retrieve()
                
                # Calculate and store capture time
                capture_time = (time.time() - start_time) * 1000
                self.capture_times.append(capture_time)
                
                # Keep only recent times
                if len(self.capture_times) > 30:
                    self.capture_times.pop(0)
            
            # Short sleep to prevent CPU overuse
            time.sleep(0.001)
            
    def read(self):
        # Return the current frame
        return self.frame
    
    def get_capture_time(self):
        # Return average capture time
        if self.capture_times:
            return sum(self.capture_times) / len(self.capture_times)
        return 0
        
    def stop(self):
        # Stop the thread and release camera
        self.stopped = True
        time.sleep(0.1)  # Give time for thread to finish
        self.stream.release()

def main():
    # Initialize YOLO model
    model = YOLO("yolov11n-face.pt")  # or use any other YOLOv11 model
    
    # Set model parameters to optimize performance
    model.conf = 0.5       # Confidence threshold
    model.iou = 0.45       # NMS IOU threshold
    model.max_det = 20     # Maximum number of detections
    
    # Additional performance settings
    torch_threads = os.cpu_count()
    if torch_threads:
        try:
            import torch
            torch.set_num_threads(torch_threads)
            print(f"Torch using {torch_threads} threads")
        except ImportError:
            pass
    
    # Start the threaded webcam stream instead of standard capture
    print("Starting threaded webcam stream...")
    vs = WebcamVideoStream(src=0, width=640, height=480).start()
    time.sleep(2.0)  # Allow camera to warm up
    
    # Create window with adjustable size
    cv2.namedWindow("YOLOv11 Face Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv11 Face Detection", 1280, 720)
    
    # FPS calculation variables
    fps = 0
    frame_count = 0
    start_time = time.time()
    fps_update_interval = 0.5  # Update FPS every half second
    fps_history = []  # Track FPS over time
    
    # Processing resolution scaling
    scale_factor = 1.0  # 1.0 = original size, 0.5 = half size, etc.
    
    # Advanced performance monitoring
    cpu_percent = 0
    last_resource_check = 0
    resource_check_interval = 1.0  # Check system resources every second
    
    print("Starting detection loop with performance monitoring...")
    print("Press 'q' to quit, '+'/'-' to adjust processing scale")
    
    while True:
        # Monitor system resources periodically
        current_time = time.time()
        if current_time - last_resource_check >= resource_check_interval:
            cpu_percent = psutil.cpu_percent()
            last_resource_check = current_time
        
        # Get frame from threaded stream
        frame = vs.read()
        if frame is None:
            print("Failed to read from camera")
            break
        
        # Get average capture time from threaded stream
        capture_time = vs.get_capture_time()
        
        # Update frame count for FPS calculation
        frame_count += 1
        elapsed_time = current_time - start_time
        
        # Record original frame dimensions for display
        orig_h, orig_w = frame.shape[:2]
        display_frame = frame.copy()
        
        # Resize for processing if scale_factor is not 1.0
        if scale_factor != 1.0:
            h, w = frame.shape[:2]
            process_w, process_h = int(w * scale_factor), int(h * scale_factor)
            process_frame = cv2.resize(frame, (process_w, process_h))
        else:
            process_frame = frame
        
        # Perform detection
        start_inference = time.time()
        results = model(process_frame)
        inference_time = (time.time() - start_inference) * 1000  # ms
        
        # Update FPS calculation every interval
        if elapsed_time > fps_update_interval:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = current_time
        
        # Draw results on display frame
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates and scale back if needed
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if scale_factor != 1.0:
                    x1 = int(x1 / scale_factor)
                    y1 = int(y1 / scale_factor)
                    x2 = int(x2 / scale_factor)
                    y2 = int(y2 / scale_factor)
                
                # Get confidence and class
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = result.names[cls]
                
                # Draw bounding box and label
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"{label} {conf:.2f}", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2)
        
        # Display performance metrics
        fps_text = f"FPS: {fps:.1f}"
        inf_text = f"Inference: {inference_time:.1f}ms"
        scale_text = f"Scale: {scale_factor:.2f}x"
        
        # Add additional performance metrics
        cap_text = f"Capture: {capture_time:.1f}ms"
        cpu_text = f"CPU: {cpu_percent}%"
        
        # Record FPS for analysis
        if elapsed_time > fps_update_interval:
            fps_history.append(fps)
            # Keep last 60 samples (30 seconds worth)
            if len(fps_history) > 60:
                fps_history.pop(0)
            
            # Calculate FPS stability
            if len(fps_history) > 5:
                fps_std = np.std(fps_history[-5:])
                fps_var_text = f"FPS Var: {fps_std:.1f}"
                cv2.putText(display_frame, fps_var_text, (10, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show performance info
        cv2.putText(display_frame, fps_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, inf_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, cap_text, (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, cpu_text, (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, scale_text, (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show the frame
        cv2.imshow("YOLOv11 Face Detection", display_frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            scale_factor = min(1.0, scale_factor + 0.1)
            print(f"Scale factor: {scale_factor:.1f}")
        elif key == ord('-') or key == ord('_'):
            scale_factor = max(0.1, scale_factor - 0.1)
            print(f"Scale factor: {scale_factor:.1f}")
    
    # Cleanup - use the thread-safe stop method
    vs.stop()
    cv2.destroyAllWindows()
    print("Detection finished")

if __name__ == "__main__":
    main()
