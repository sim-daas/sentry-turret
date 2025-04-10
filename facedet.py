import cv2
from ultralytics import YOLO
import serial
import time
import math
from collections import defaultdict
import numpy as np


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


def select_face_to_track(frame, model):
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
        print(f"Selected object: {selected_object['label']} (class {selected_object['cls']})")
    
    return selected_object


def logic(boxes, selected_object, ids, prev_angles=None):
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
        return prev_angles
    
    if selected_object:
        selected_object.update(target_box)
    
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
        tilt_angle = 90 + math.degrees(ang)
    else:
        cy = 240 - cy
        ang = math.atan2(yth*cy, 240)
        tilt_angle = 90 - math.degrees(ang)
    
    pan_angle = max(10, min(170, pan_angle))
    tilt_angle = max(10, min(170, tilt_angle))
    
    smooth_factor = 0.3
    
    pan_angle = prev_angles[0] + smooth_factor * (pan_angle - prev_angles[0])
    tilt_angle = prev_angles[1] + smooth_factor * (tilt_angle - prev_angles[1])
    
    return [pan_angle, tilt_angle]


def draw_boxes(frame, boxes, history):
    for box in boxes:
        x1, y1, x2, y2 = box['coords']
        lbl = f"{box['label']} {box['conf']:.2f}"
        
        clr = (0, 255, 0)
        if box.get('tid') is not None:
            tid = box['tid']
            c_id = tid * 5 % 256
            clr = (c_id, 255, 255 - c_id)
            
            lbl = f"ID:{tid} " + lbl
            
            if tid in history and len(history[tid]) > 1:
                pts = np.array(history[tid], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=False, color=(230, 230, 230), thickness=2)
        
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


def send_servo_command(ser, angles):
    if not ser or not ser.is_open:
        print("Serial connection not available")
        return False
        
    try:
        pan_angle, tilt_angle = angles
        
        if 10 <= pan_angle <= 170 and 10 <= tilt_angle <= 170:
            cmd = f"{int(pan_angle)},{int(tilt_angle)}\n" 
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
    
    xth = math.tan(math.radians(30))
    yth = math.tan(40)
    
    model = YOLO("yolov11n-face.pt")
    
    last_cmd_time = 0
    cmd_interval = 0.001
    prev_angles = [90, 90]
    
    track_history = defaultdict(lambda: [])
    max_history = 50
    
    fps = 0
    frame_count = 0
    start_time = time.time()
    fps_update_interval = 0.5
    
    flip_frame = False
    
    servo_connection = setup_servo_connection()
    
    cap = cv2.VideoCapture('/dev/video0')
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to capture first frame from camera")
        return
    
    if flip_frame:
        first_frame = cv2.flip(first_frame, 1)
    
    selected_object = select_face_to_track(first_frame, model)
    
    tracking_enabled = True
    
    while cap.isOpened():
        ret, frame = cap.read() 
        if not ret:
            break
        
        if flip_frame:
            frame = cv2.flip(frame, 1)
        
        frame_count += 1
        elapsed_time = time.time() - start_time
        
        if elapsed_time > fps_update_interval:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        boxes, ids = get_face_boxes(frame, model, tracking=tracking_enabled)
        
        for box in boxes:
            if box.get('tid') is not None:
                tid = box['tid']
                x1, y1, x2, y2 = box['coords']
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                track_history[tid].append((cx, cy))
                if len(track_history[tid]) > max_history:
                    track_history[tid].pop(0)
        
        frame = draw_boxes(frame, boxes, track_history)
        
        current_time = time.time()
        if boxes and (current_time - last_cmd_time) >= cmd_interval:
            angles = logic(boxes, selected_object, ids, prev_angles)
            
            print(f"Sending angles: Pan={angles[0]:.1f}, Tilt={angles[1]:.1f}")
            send_servo_command(servo_connection, angles)
            prev_angles = angles
            last_cmd_time = current_time
        
        status = "Tracking: ON" if tracking_enabled else "Tracking: OFF"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 0, 255), 2)
        
        fps_txt = f"FPS: {fps:.1f}"
        fps_sz = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        fps_x = frame.shape[1] - fps_sz[0] - 10
        cv2.putText(frame, fps_txt, (fps_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 255), 2)
        
        cv2.imshow("Object Tracking", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            tracking_enabled = not tracking_enabled
            print(f"Tracking {'enabled' if tracking_enabled else 'disabled'}")
        elif key == ord('r'):
            selected_object = None
            print("Reset selected object - will track first detected object")
        elif key == ord('f'):
            flip_frame = not flip_frame
            print(f"Frame flipping {'enabled' if flip_frame else 'disabled'}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    if servo_connection and servo_connection.is_open:
        servo_connection.close()
        print("Serial connection closed.")


if __name__ == "__main__":
    main()
