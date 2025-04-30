import cv2
from ultralytics import YOLO
import serial
import time
import math
from collections import defaultdict
import numpy as np
import face_recognition
import pickle
import os

# Constants for visibility criteria
MIN_FACE_AREA = 2500  # Minimum pixel area (e.g., 50x50)
BOUNDARY_WIDTH = 100   # Pixels from edge

def load_face_db():
    db_dir = "face_db"
    emb_file = os.path.join(db_dir, "embeddings.pkl")
    
    if os.path.exists(emb_file):
        with open(emb_file, "rb") as f:
            return pickle.load(f)
    return {}


def get_face_boxes(frame, model, tracking=False):
    if tracking:
        res = model.track(frame, persist=True)
    else:
        res = model(frame)
    
    boxes = []
    ids = []
    
    for r in res:
        if r.boxes:
            for i, bx in enumerate(r.boxes):
                x1, y1, x2, y2 = map(int, bx.xyxy[0])
                cf = float(bx.conf[0])
                cl = int(bx.cls[0])
                lb = r.names[cl]
                
                tid = None
                if hasattr(bx, 'id') and bx.id is not None:
                    tid = int(bx.id[0])
                    ids.append(tid)
                
                boxes.append({
                    'coords': (x1, y1, x2, y2),
                    'conf': cf,
                    'cls': cl,
                    'label': lb,
                    'tid': tid
                })
    
    return boxes, ids


def is_face_visible(box, frame_shape):
    """Checks if a face meets visibility criteria."""
    x1, y1, x2, y2 = box['coords']
    h, w = frame_shape[:2]

    # Check area
    area = (x2 - x1) * (y2 - y1)
    if area < MIN_FACE_AREA:
        # print(f"Face too small: {area} < {MIN_FACE_AREA}")
        return False

    # Check centroid position
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    if not (BOUNDARY_WIDTH <= cx <= w - BOUNDARY_WIDTH and
            BOUNDARY_WIDTH <= cy <= h - BOUNDARY_WIDTH):
        # print(f"Face too close to edge: cx={cx}, cy={cy}")
        return False

    return True


def recognize_face(frame, box, known_embs):
    """Recognizes a face and returns name and known status."""
    x1, y1, x2, y2 = box['coords']

    face_img = frame[y1:y2, x1:x2]
    if face_img.shape[0] < 10 or face_img.shape[1] < 10:
        return "Unknown", False # Too small to process reliably

    try:
        # Convert BGR (OpenCV default) to RGB (face_recognition default)
        rgb_face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        h, w = rgb_face_img.shape[:2]
        face_loc = (0, w, h, 0) # Bbox relative to the cropped face image
        
        # Use RGB image for encoding
        encodings = face_recognition.face_encodings(rgb_face_img, [face_loc])
        if not encodings:
            # print("Could not generate encoding for face.")
            return "Unknown", False
        face_emb = encodings[0]
        
        min_dist = float('inf')
        best_name = "Unknown"
        
        for name, embs in known_embs.items():
            for emb in embs:
                dist = np.linalg.norm(face_emb - emb)
                # Using a threshold of 0.6 for matching
                if dist < min_dist and dist < 0.6:
                    min_dist = dist
                    best_name = name
        
        is_known = best_name != "Unknown"
        # print(f"Recognition result: {best_name}, Known: {is_known}, Dist: {min_dist if is_known else 'N/A'}")
        return best_name, is_known
    except Exception as e:
        print(f"Recognition error: {e}")
        return "Unknown", False


def select_face_to_track(frame, model, known_embs, known_faces):
    print("Select a face/object to track. Press SPACE to confirm selection or ESC to track first detected face.")
    
    boxes, _ = get_face_boxes(frame, model)
    if not boxes:
        print("No objects detected in first frame. Will track first object when detected.")
        return None
    
    sel_frame = frame.copy()
    sel_idx = 0
    
    def update_selection():
        tmp_frame = frame.copy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box['coords']
            clr = (0, 0, 255) if i == sel_idx else (0, 255, 0)
            thk = 3 if i == sel_idx else 2
            cv2.rectangle(tmp_frame, (x1, y1), (x2, y2), clr, thk)
            
            lbl = f"{i+1}: {box['label']} {box['conf']:.2f}"
            cv2.putText(tmp_frame, lbl, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)
        
        inst = "Use arrow keys to select, SPACE to confirm, ESC to auto-select"
        cv2.putText(tmp_frame, inst, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return tmp_frame
    
    sel_frame = update_selection()
    cv2.imshow("Select Object to Track", sel_frame)
    
    while True:
        key = cv2.waitKey(0)
        
        if key == 27:
            selected_object = boxes[0] if boxes else None
            break
            
        elif key == 32:
            selected_object = boxes[sel_idx] if boxes else None
            break
            
        elif key == 83 or key == 100:
            sel_idx = (sel_idx + 1) % len(boxes)
            sel_frame = update_selection()
            cv2.imshow("Select Object to Track", sel_frame)
            
        elif key == 81 or key == 97:
            sel_idx = (sel_idx - 1) % len(boxes)
            sel_frame = update_selection()
            cv2.imshow("Select Object to Track", sel_frame)
    
    cv2.destroyWindow("Select Object to Track")
    
    if selected_object:
        print(f"Selected object: {selected_object['label']} (class {selected_object['cls']}) with TID: {selected_object.get('tid')}")

        # Immediately attempt recognition on the selected face if it has a TID
        tid = selected_object.get('tid')
        if tid is not None:
            # Initialize status in known_faces
            if tid not in known_faces:
                 known_faces[tid] = {'name': 'Unknown', 'is_known': False, 'status': 'pending', 'last_recog_attempt': 0}

            # Check visibility before immediate recognition
            if is_face_visible(selected_object, frame.shape):
                print("Attempting immediate recognition for selected face...")
                name, is_known = recognize_face(frame, selected_object, known_embs)
                print(f"Immediate recognition result: {name}, Known: {is_known}")
                known_faces[tid].update({
                    'name': name,
                    'is_known': is_known,
                    'status': 'recognized',
                    'last_recog_attempt': time.time()
                })
                # Add recognition info directly to selected_object for potential immediate use
                selected_object['name'] = name
                selected_object['is_known'] = is_known
            else:
                print("Selected face not clearly visible for immediate recognition, will attempt later.")
                # Keep status as 'pending'
        else:
             print("Selected object has no tracking ID, cannot perform persistent recognition.")


    return selected_object


def logic(boxes, selected_object, ids, known_faces, prev_angles=None):
    global xth, yth
    
    if prev_angles is None:
        prev_angles = [90, 90]
    
    target_box = None
    
    if selected_object and selected_object.get('tid') is not None:
        for box in boxes:
            if box.get('tid') == selected_object['tid']:
                target_box = box
                break
    
    if not target_box and boxes:
        target_box = boxes[0]
    
    if not target_box:
        return prev_angles, None # Return angles and no target tid
    
    # Update the selected_object's data with the latest from target_box if they match
    if selected_object and target_box.get('tid') == selected_object.get('tid'):
         # Preserve recognition results if already present in selected_object
        current_name = selected_object.get('name', 'Unknown')
        current_is_known = selected_object.get('is_known', False)
        selected_object.update(target_box)
        # Restore recognition results if overwritten by basic box update
        if 'name' not in selected_object: selected_object['name'] = current_name
        if 'is_known' not in selected_object: selected_object['is_known'] = current_is_known

    x1, y1, x2, y2 = target_box['coords']
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    
    if cx >= 320:
        cx -= 320
        ang = math.atan2(xth*cx, 320)
        pan_angle = 90 - math.degrees(ang)
    else:
        cx = 320 - cx
        ang = math.atan2(xth*cx, 320)
        pan_angle = 90 + math.degrees(ang)
    
    if cy >= 240:
        cy -= 240
        ang = math.atan2(yth*cy, 240)
        tilt_angle = 90 - math.degrees(ang)
    else:
        cy = 240 - cy
        ang = math.atan2(yth*cy, 240)
        tilt_angle = 90 + math.degrees(ang)
    
    pan_angle = max(10, min(170, pan_angle))
    tilt_angle = max(10, min(170, tilt_angle))
    
    smooth_factor = 0.3
    
    pan_angle = prev_angles[0] + smooth_factor * (pan_angle - prev_angles[0])
    tilt_angle = prev_angles[1] + smooth_factor * (tilt_angle - prev_angles[1])
    
    # Return calculated angles and the tid of the box being tracked
    return [pan_angle, tilt_angle], target_box.get('tid')


def draw_boxes(frame, boxes, history, known_faces):
    for box in boxes:
        x1, y1, x2, y2 = box['coords']
        tid = box.get('tid')
        lbl = ""
        clr = (0, 255, 0) # Default green

        if tid is not None:
            # Get status from known_faces
            face_info = known_faces.get(tid)

            if face_info:
                status = face_info['status']
                name = face_info['name']
                is_known = face_info['is_known']

                if status == 'recognized':
                    lbl = f"{name} ({tid})"
                    clr = (0, 255, 0) if is_known else (0, 0, 255) # Green if known, Red if unknown
                elif status == 'pending':
                    lbl = f"ID:{tid} (Pending)"
                    clr = (255, 165, 0) # Orange for pending recognition
                else: # Should not happen, but fallback
                    lbl = f"ID:{tid} (Status?)"
                    clr = (255, 255, 0) # Cyan for unknown status
            else:
                # TID exists but not yet in known_faces (should be handled in main loop)
                lbl = f"ID:{tid} (New)"
                clr = (255, 255, 255) # White for newly detected

            # Draw tracking history
            if tid in history and len(history[tid]) > 1:
                pts = np.array(history[tid], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=False, color=(230, 230, 230), thickness=2)
        else:
            # Box without tracking ID (shouldn't happen if tracking is enabled)
            lbl = f"{box['label']} {box['conf']:.2f}"
            clr = (0, 255, 255) # Yellow for non-tracked detections

        cv2.rectangle(frame, (x1, y1), (x2, y2), clr, 2)
        cv2.putText(frame, lbl, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, clr, 2)
    
    return frame


def setup_servo_connection(port='/dev/ttyACM0', baud=9600):
    try:
        ser = serial.Serial(port, baud, timeout=1)
        print(f"Connected to Arduino on {port}")
        time.sleep(0.5)
        ser.reset_input_buffer()
        return ser
    except serial.SerialException as e:
        print(f"Error opening serial port:{e}")
        print("Make sure Arduino is connected and the port is correct.")
        return None


def send_servo_command(ser, angles, laser_en):
    if not ser or not ser.is_open:
        print("Serial connection not available")
        return False
        
    try:
        pan_angle, tilt_angle = angles
        
        if 10 <= pan_angle <= 170 and 10 <= tilt_angle <= 170:
            cmd = f"{int(pan_angle)},{int(tilt_angle)},{int(laser_en)}\n"
            ser.write(cmd.encode())
            time.sleep(0.001)
            return True
        else:
            print(f"Invalid angle values: Pan={pan_angle}, Tilt={tilt_angle}. Must be between 10 and 170.")
            return False
    except Exception as e:
        print(f"Error sending command: {e}")
        return False


def main():
    global xth, yth

    xth = math.tan(math.radians(35))
    yth = math.tan(math.radians(25))

    known_embs = load_face_db()
    # known_faces stores status per track ID:
    # { tid: {'name': '...', 'is_known': True/False, 'status': 'pending'/'recognized', 'last_recog_attempt': timestamp} }
    known_faces = {}

    model = YOLO("yolov11l-face.pt") # Make sure this model detects faces

    last_cmd_time = 0
    cmd_interval = 0.001 # Interval for sending servo commands
    prev_angles = [90, 90]
    recog_interval = 1.0 # Minimum interval between recognition attempts *per face*

    track_history = defaultdict(lambda: [])
    max_history = 50

    fps = 0
    frame_count = 0
    start_time = time.time()
    fps_update_interval = 0.5

    flip_frame = False

    servo_connection = setup_servo_connection()

    cap = cv2.VideoCapture('/dev/video0') # Or your camera index/path
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("Failed to capture first frame from camera")
        if servo_connection: servo_connection.close()
        cap.release()
        return

    if flip_frame:
        first_frame = cv2.flip(first_frame, 1)

    # Select initial object/face to track
    selected_object = select_face_to_track(first_frame, model, known_embs, known_faces)

    tracking_enabled = True # Start with tracking enabled

    # Create the display window beforehand
    cv2.namedWindow("Face Recognition Tracking", cv2.WINDOW_AUTOSIZE)

    while cap.isOpened():
        ret, frame = cap.read() 
        if not ret:
            break
        
        if flip_frame:
            frame = cv2.flip(frame, 1)
        
        frame_height, frame_width = frame.shape[:2]
        current_time = time.time()

        # --- FPS Calculation ---
        frame_count += 1
        elapsed_time = current_time - start_time
        if elapsed_time > fps_update_interval:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        # --- Object Detection and Tracking ---
        boxes, ids = get_face_boxes(frame, model, tracking=tracking_enabled)

        # --- Face Recognition Logic ---
        for box in boxes:
            tid = box.get('tid')
            if tid is not None:
                # Initialize status for new tracks
                if tid not in known_faces:
                    known_faces[tid] = {'name': 'Unknown', 'is_known': False, 'status': 'pending', 'last_recog_attempt': 0}

                # Attempt recognition only if status is pending and criteria met
                face_info = known_faces[tid]
                if face_info['status'] == 'pending':
                    if is_face_visible(box, frame.shape):
                        if current_time - face_info['last_recog_attempt'] >= recog_interval:
                            # print(f"Attempting recognition for visible TID: {tid}")
                            name, is_known = recognize_face(frame, box, known_embs)
                            known_faces[tid].update({
                                'name': name,
                                'is_known': is_known,
                                'status': 'recognized',
                                'last_recog_attempt': current_time
                            })
                            # Update selected_object if it matches this tid
                            if selected_object and selected_object.get('tid') == tid:
                                selected_object['name'] = name
                                selected_object['is_known'] = is_known
                                # print(f"Updated selected_object ({tid}) recognition: {name}, Known: {is_known}")
                        # else:
                            # print(f"TID {tid} visible but waiting for recog interval.")
                    # else:
                        # print(f"TID {tid} not visible enough for recognition.")


        # --- Update Track History ---
        active_tids = set()
        for box in boxes:
             tid = box.get('tid')
             if tid is not None:
                 active_tids.add(tid)
                 x1, y1, x2, y2 = box['coords']
                 cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                 track_history[tid].append((cx, cy))
                 if len(track_history[tid]) > max_history:
                     track_history[tid].pop(0)

        # --- Clean up old tracks ---
        expired_tids = set(track_history.keys()) - active_tids
        for tid in expired_tids:
            # print(f"Removing expired track ID: {tid}")
            del track_history[tid]
            if tid in known_faces:
                del known_faces[tid]
            # If the expired track was the selected object, reset selection
            if selected_object and selected_object.get('tid') == tid:
                print(f"Selected object (TID: {tid}) lost.")
                selected_object = None # Reset selection

        # --- Drawing ---
        frame = draw_boxes(frame, boxes, track_history, known_faces)

        # --- Servo Control Logic ---
        target_tid = None
        laser_en = 0 # Default laser OFF

        if boxes and (current_time - last_cmd_time) >= cmd_interval:
            # Pass known_faces to logic function if needed, or determine target tid here
            angles, target_tid = logic(boxes, selected_object, ids, known_faces, prev_angles)

            if target_tid is not None:
                # Determine laser state based on the tracked face's status
                if target_tid in known_faces:
                    target_info = known_faces[target_tid]
                    # Laser ON only if recognized as Unknown
                    if target_info['status'] == 'recognized' and not target_info['is_known']:
                        laser_en = 1
                    else: # Pending recognition or Known face
                        laser_en = 0
                else:
                    # Should not happen if logic is correct, but default to OFF
                    laser_en = 0

                # Print the angles and laser status before sending
                print(f"Sending angles: Pan={angles[0]:.1f}, Tilt={angles[1]:.1f}, Laser={laser_en}")
                send_servo_command(servo_connection, angles, laser_en)
                prev_angles = angles
                last_cmd_time = current_time
            # else: No target found, don't send command / send neutral? (Current logic keeps prev_angles)


        # --- Display FPS ---
        fps_txt = f"FPS: {fps:.1f}"
        fps_sz = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        fps_x = frame.shape[1] - fps_sz[0] - 10
        cv2.putText(frame, fps_txt, (fps_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 255), 2)

        # --- Display Frame ---
        # The window created above will be used here
        cv2.imshow("Face Recognition Tracking", frame)

        # --- Key Controls ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            tracking_enabled = not tracking_enabled
            print(f"Tracking {'enabled' if tracking_enabled else 'disabled'}")
            if not tracking_enabled: # Clear history/status if tracking is disabled
                 track_history.clear()
                 known_faces.clear()
                 selected_object = None
        elif key == ord('r'):
            selected_object = None
            print("Reset selected object - will track first detected object")
            # Don't clear known_faces here, just the selection
        elif key == ord('f'):
            flip_frame = not flip_frame
            print(f"Frame flipping {'enabled' if flip_frame else 'disabled'}")
        elif key == ord('s'): # Manual re-selection
             print("Attempting re-selection...")
             # Need to capture a frame *here* for selection
             temp_ret, temp_frame = cap.read()
             if temp_ret:
                 if flip_frame: temp_frame = cv2.flip(temp_frame, 1)
                 # Pass current known_faces for potential immediate recognition update
                 new_selection = select_face_to_track(temp_frame, model, known_embs, known_faces)
                 if new_selection:
                     selected_object = new_selection
                     print("New object selected.")
                 else:
                     print("Selection cancelled or failed.")
             else:
                 print("Failed to capture frame for re-selection.")


    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()

    if servo_connection and servo_connection.is_open:
        # Optionally send servos to neutral position
        # send_servo_command(servo_connection, [90, 90], 0)
        # time.sleep(0.1)
        servo_connection.close()
        print("Serial connection closed.")


if __name__ == "__main__":
    main()
