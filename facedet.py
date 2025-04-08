import cv2
from ultralytics import YOLO
import serial
import time
import math
from collections import defaultdict
import numpy as np


def get_face_boxes(frame, model, tracking=False):
    # Performance improvement: Reduce processing resolution
    # Only process a smaller frame for detection (maintain display resolution)
    height, width = frame.shape[:2]
    if width > 640:  # Only resize if frame is large
        scale_factor = 640 / width
        process_frame = cv2.resize(frame, (640, int(height * scale_factor)))
    else:
        process_frame = frame
        
    if tracking:
        results = model.track(process_frame, persist=True)
    else:
        results = model(process_frame)
    
    # Scale boxes back to original size if we resized
    scale_x = width / process_frame.shape[1] if process_frame.shape[1] != width else 1.0
    scale_y = height / process_frame.shape[0] if process_frame.shape[0] != height else 1.0
    
    boxes = []
    track_ids = []
    
    for result in results:
        if result.boxes:
            for i, box in enumerate(result.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = result.names[cls]
                
                # Get track ID if available
                track_id = None
                if hasattr(box, 'id') and box.id is not None:
                    track_id = int(box.id[0])
                    track_ids.append(track_id)
                
                boxes.append({
                    'coords': (x1, y1, x2, y2),
                    'confidence': conf,
                    'class': cls,
                    'label': label,
                    'track_id': track_id
                })
    
    return boxes, track_ids


def select_face_to_track(frame, model):
    """Allow user to select a face/object to track from the first frame"""
    print("Select a face/object to track. Press SPACE to confirm selection or ESC to track first detected face.")
    
    # Get initial detection
    boxes, _ = get_face_boxes(frame, model)
    if not boxes:
        print("No objects detected in first frame. Will track first object when detected.")
        return None
    
    # Draw boxes on a copy of the frame
    selection_frame = frame.copy()
    selected_index = 0  # Start with first detected object
    
    # Function to update the visual selection
    def update_selection():
        temp_frame = frame.copy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box['coords']
            color = (0, 0, 255) if i == selected_index else (0, 255, 0)  # Red for selected, green for others
            thickness = 3 if i == selected_index else 2
            cv2.rectangle(temp_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Add label with index for selection
            label = f"{i+1}: {box['label']} {box['confidence']:.2f}"
            cv2.putText(temp_frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        instructions = "Use arrow keys to select, SPACE to confirm, ESC to auto-select"
        cv2.putText(temp_frame, instructions, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return temp_frame
    
    # Show selection interface
    selection_frame = update_selection()
    cv2.imshow("Select Object to Track", selection_frame)
    
    # Wait for key press to select object
    while True:
        key = cv2.waitKey(0)
        
        # ESC key - use first object
        if key == 27:
            selected_object = boxes[0] if boxes else None
            break
            
        # SPACE key - confirm current selection
        elif key == 32:
            selected_object = boxes[selected_index] if boxes else None
            break
            
        # Right arrow - next object
        elif key == 83 or key == 100:  # Right arrow or 'd'
            selected_index = (selected_index + 1) % len(boxes)
            selection_frame = update_selection()
            cv2.imshow("Select Object to Track", selection_frame)
            
        # Left arrow - previous object
        elif key == 81 or key == 97:  # Left arrow or 'a'
            selected_index = (selected_index - 1) % len(boxes)
            selection_frame = update_selection()
            cv2.imshow("Select Object to Track", selection_frame)
    
    cv2.destroyWindow("Select Object to Track")
    
    if selected_object:
        print(f"Selected object: {selected_object['label']} (class {selected_object['class']})")
    
    return selected_object


def logic(boxes, selected_object, track_ids, prev_angles=None):
    """Calculate servo angles based on selected object position with smoothing."""
    global xtheta, ytheta
    
    # Initialize previous angles if not provided
    if prev_angles is None:
        prev_angles = [90, 90]  # Default angles for pan and tilt
    
    # If we have a selected object with a track_id, find it in the current boxes
    target_box = None
    
    if selected_object and selected_object.get('track_id') is not None:
        # Look for the box with matching track_id
        for box in boxes:
            if box.get('track_id') == selected_object['track_id']:
                target_box = box
                break
    
    # If no specific tracked object or it's not found, use the first box (default behavior)
    if not target_box and boxes:
        target_box = boxes[0]
    
    # If no boxes, can't calculate angles
    if not target_box:
        return prev_angles
    
    # Update the selected object to maintain tracking
    if selected_object:
        selected_object.update(target_box)
    
    x1, y1, x2, y2 = target_box['coords']
    cent_x = (x1 + x2) / 2    # X center for pan (horizontal)
    cent_y = (y1 + y2) / 2    # Y center for tilt (vertical)
    
    # Calculate pan angle (horizontal)
    if cent_x >= 320:
        cent_x -= 320
        ang = math.atan2(xtheta*cent_x, 320)
        new_angle_pan = 90 + math.degrees(ang)
    else:
        cent_x = 320 - cent_x
        ang = math.atan2(xtheta*cent_x, 320)
        new_angle_pan = 90 - math.degrees(ang)
    
    # Calculate tilt angle (vertical)
    if cent_y >= 240:  # Assuming 480x640 resolution, vertical center is 240
        cent_y -= 240
        ang = math.atan2(ytheta*cent_y, 240)
        new_angle_tilt = 90 + math.degrees(ang)
    else:
        cent_y = 240 - cent_y
        ang = math.atan2(ytheta*cent_y, 240)
        new_angle_tilt = 90 - math.degrees(ang)
    
    # Constrain angles within servo limits
    new_angle_pan = max(10, min(170, new_angle_pan))
    new_angle_tilt = max(10, min(170, new_angle_tilt))
    
    # Apply smoothing if we have previous angles
    smoothing_factor = 0.3  # Lower values = more smoothing
    
    # Smooth pan angle
    new_angle_pan = prev_angles[0] + smoothing_factor * (new_angle_pan - prev_angles[0])
    
    # Smooth tilt angle
    new_angle_tilt = prev_angles[1] + smoothing_factor * (new_angle_tilt - prev_angles[1])
    
    return [new_angle_pan, new_angle_tilt]


def draw_boxes(frame, boxes, track_history):
    """Draw bounding boxes, labels, and tracking lines on the frame."""
    # Performance optimization: Only draw recent history points (30 max)
    # and skip drawing every other point to reduce rendering load
    history_limit = 30
    
    for box in boxes:
        x1, y1, x2, y2 = box['coords']
        label = f"{box['label']} {box['confidence']:.2f}"
        
        # Different color for tracked object
        color = (0, 255, 0)  # Green for detected objects
        if box.get('track_id') is not None:
            # Different color based on track_id for visual distinction
            track_id = box['track_id']
            # Generate a unique color based on track_id
            color_id = track_id * 5 % 256
            color = (color_id, 255, 255 - color_id)
            
            # Add track_id to label
            label = f"ID:{track_id} " + label
            
            # Draw tracking line if history exists (optimized)
            if track_id in track_history and len(track_history[track_id]) > 1:
                # Only use the most recent points (up to history_limit)
                recent_points = track_history[track_id][-history_limit:]
                # Skip every other point to reduce drawing overhead
                sparse_points = recent_points[::2]  
                if len(sparse_points) > 1:
                    points = np.array(sparse_points, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, color, 2)
    
    return frame


def setup_servo_connection(port='/dev/ttyACM0', baud_rate=9600):
    """Establish a serial connection to the Arduino controlling the servos."""
    try:
        ser = serial.Serial(port, baud_rate, timeout=1)
        print(f"Connected to Arduino on {port}")
        time.sleep(0.5)
        ser.reset_input_buffer()
        return ser
    except serial.SerialException as e:
        print(f"Error opening serial port:{e}")
        print("Make sure Arduino is connected and the port is correct.")
        return None


def send_servo_command(ser, angles):
    """Send servo position commands to the Arduino.
    
    Args:
        ser: Serial connection object
        angles: List containing [pan_angle, tilt_angle]
    """
    if not ser or not ser.is_open:
        print("Serial connection not available")
        return False
        
    try:
        pan_angle, tilt_angle = angles
        
        # Validate both angles
        if 10 <= pan_angle <= 170 and 10 <= tilt_angle <= 170:
            # Format command with both angles, separated by comma, with newline
            command = f"{int(pan_angle)},{int(tilt_angle)}\n" 
            ser.write(command.encode())
            # Wait a bit for the command to be processed
            time.sleep(0.001)
            return True
        else:
            print(f"Invalid angle values: Pan={pan_angle}, Tilt={tilt_angle}. Must be between 10 and 170.")
            return False
    except Exception as e:
        print(f"Error sending command: {e}")
        return False


def main():
    global xtheta, ytheta
    
    xtheta = math.tan(math.radians(30))
    ytheta = math.tan(40)
    
    model = YOLO("yolov11n-face.pt")
    
    last_command_time = 0
    command_interval = 0.03  # 30ms interval is sufficient (33Hz)
    prev_angles = [90, 90]  # Initialize with default [pan, tilt] angles
    
    # Track history for visualization
    track_history = defaultdict(lambda: [])
    max_history = 40  # 40 points is more than enough for visualization
    
    # FPS calculation variables
    fps = 0
    frame_count = 0
    start_time = time.time()
    fps_update_interval = 0.5  # Update FPS display every half second
    
    # Frame flip control
    flip_frame = True  # Default: no flipping
    
    servo_connection = setup_servo_connection()
    
    cap = cv2.VideoCapture('/dev/video0')
    
    # Optionally reduce resolution if needed for performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Get the first frame for selection
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to capture first frame from camera")
        return
    
    # Apply flip if enabled
    if flip_frame:
        first_frame = cv2.flip(first_frame, 1)  # 1 for horizontal flip
    
    # Allow user to select a face/object to track
    selected_object = select_face_to_track(first_frame, model)
    
    # Enable tracking
    tracking_enabled = True
    
    # Enable non-maximum suppression for the model to reduce duplicate detections
    model.conf = 0.5       # Higher confidence threshold
    model.iou = 0.45       # IOU threshold
    model.max_det = 5      # Maximum detections per frame - we only need a few
    
    while cap.isOpened():
        ret, frame = cap.read() 
        if not ret:
            break
        
        # Apply flip if enabled
        if flip_frame:
            frame = cv2.flip(frame, 1)  # 1 for horizontal flip
        
        # Update frame count for FPS calculation
        frame_count += 1
        elapsed_time = time.time() - start_time
        
        # Update FPS calculation every interval
        if elapsed_time > fps_update_interval:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        # Only run tracking when necessary - toggle between detection/tracking
        # Detection is faster when you don't need to track IDs
        should_track = tracking_enabled and (frame_count % 2 == 0)  # Only track every other frame
        boxes, track_ids = get_face_boxes(frame, model, tracking=should_track)
        
        # Only update track history every other frame to reduce overhead
        if frame_count % 2 == 0:
            for box in boxes:
                if box.get('track_id') is not None:
                    track_id = box['track_id']
                    x1, y1, x2, y2 = box['coords']
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    track_history[track_id].append((center_x, center_y))
                    # Use a smaller max history
                    if len(track_history[track_id]) > max_history:
                        track_history[track_id].pop(0)
        
        # Draw all detections and tracks
        frame = draw_boxes(frame, boxes, track_history)
        
        current_time = time.time()
        if boxes and (current_time - last_command_time) >= command_interval:
            angles = logic(boxes, selected_object, track_ids, prev_angles)
            
            # Send the angles to the servos
            print(f"Sending angles: Pan={angles[0]:.1f}, Tilt={angles[1]:.1f}")
            send_servo_command(servo_connection, angles)
            prev_angles = angles
            last_command_time = current_time
        
        # Display tracking status and FPS
        status = "Tracking: ON" if tracking_enabled else "Tracking: OFF"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 0, 255), 2)
        
        # Display FPS in top right corner
        fps_text = f"FPS: {fps:.1f}"
        fps_text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        fps_x = frame.shape[1] - fps_text_size[0] - 10  # 10 pixels from right edge
        cv2.putText(frame, fps_text, (fps_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 255), 2)  # Yellow color for visibility
        
        cv2.imshow("Object Tracking", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):  # Toggle tracking
            tracking_enabled = not tracking_enabled
            print(f"Tracking {'enabled' if tracking_enabled else 'disabled'}")
        elif key == ord('r'):  # Reset selected object
            selected_object = None
            print("Reset selected object - will track first detected object")
        elif key == ord('f'):  # Toggle frame flipping
            flip_frame = not flip_frame
            print(f"Frame flipping {'enabled' if flip_frame else 'disabled'}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    if servo_connection and servo_connection.is_open:
        servo_connection.close()
        print("Serial connection closed.")


if __name__ == "__main__":
    main()
