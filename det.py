import cv2
import time
from ultralytics import YOLO
import numpy as np

def main():
    # Initialize YOLO model
    model = YOLO("yolov11s-face.pt")  # or use any other YOLOv11 model
    
    # Set model parameters to optimize performance
    model.conf = 0.5       # Confidence threshold
    model.iou = 0.45       # NMS IOU threshold
    model.max_det = 20     # Maximum number of detections
    
    # Open webcam
    cap = cv2.VideoCapture(0)  # Change to appropriate camera index if needed
    
    # FPS calculation variables
    fps = 0
    frame_count = 0
    start_time = time.time()
    fps_update_interval = 0.5  # Update FPS every half second
    
    # Create window with adjustable size
    cv2.namedWindow("YOLOv11 Face Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv11 Face Detection", 1280, 720)
    
    # Processing resolution scaling
    scale_factor = 1.0  # 1.0 = original size, 0.5 = half size, etc.
    
    print("Starting detection loop...")
    print("Press 'q' to quit, '+'/'-' to adjust processing scale")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera")
            break
        
        # Update frame count for FPS calculation
        frame_count += 1
        current_time = time.time()
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
        
        cv2.putText(display_frame, fps_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, inf_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, scale_text, (10, 90), 
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
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Detection finished")

if __name__ == "__main__":
    main()
