import cv2
from ultralytics import YOLO
import serial
import time
import math


def get_face_boxes(frame, model):
    results = model(frame)
    boxes = []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = result.names[cls]
            boxes.append({
                'coords': (x1, y1, x2, y2),
                'confidence': conf,
                'class': cls,
                'label': label
            })
    
    return boxes

def logic(boxes):
    global xtheta, ytheta
    box = boxes[0]
    x1, y1, x2, y2 = box['coords']
    cent = (x1 + x2) / 2
    if cent >= 320:
        cent -= 320
        ang = math.atan2(xtheta*cent, 320)
        
        return 90+math.degrees(ang)
    else:
        cent = 320 - cent
        ang = math.atan2(xtheta*cent, 320)

        return 90 - math.degrees(ang)
    
    
def draw_boxes(frame, boxes):
    """Draw bounding boxes and labels on the frame."""
    for box in boxes:
        x1, y1, x2, y2 = box['coords']
        label = f"{box['label']} {box['confidence']:.2f}"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 2)
    
    return frame

def setup_servo_connection(port='/dev/ttyACM0', baud_rate=9600):
    """Establish a serial connection to the Arduino controlling the servos."""
    try:
        ser = serial.Serial(port, baud_rate, timeout=1)
        print(f"Connected to Arduino on {port}")
        time.sleep(2)
        ser.reset_input_buffer()
        return ser
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        print("Make sure Arduino is connected and the port is correct.")
        return None

def send_servo_command(ser, angle):
    """Send a servo position command to the Arduino.
    """
    if not ser or not ser.is_open:
        print("Serial connection not available")
        return False
        
    try:
        if 10 <= angle <= 170:
            command = f"{int(angle)}\n" 
            ser.write(command.encode())
            time.sleep(0.05)
            return True
        else:
            print(f"Invalid angle value: {angle}. Must be between 10 and 170.")
            return False
    except Exception as e:
        print(f"Error sending command: {e}")
        return False

model = YOLO("yolov11s-face.pt")   

xtheta = math.tan(math.radians(30))
ytheta = math.tan(22)

last_command_time = 0
command_interval = 0.05   

servo_connection = setup_servo_connection()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()   
    if not ret:
        break

    face_boxes = get_face_boxes(frame, model)
    
    frame = draw_boxes(frame, face_boxes)
    
    current_time = time.time()
    if face_boxes and (current_time - last_command_time) >= command_interval:
        angle = logic(face_boxes)
        
        angle_to_send = int(round(angle))
        
        print(f"Sending angle: {angle_to_send}")
        send_servo_command(servo_connection, angle_to_send)
        last_command_time = current_time
     
    cv2.imshow("YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if 'servo_connection' in locals() and servo_connection and servo_connection.is_open:
    servo_connection.close()
    print("Serial connection closed.")
