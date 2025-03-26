import cv2
from ultralytics import YOLO
import serial
import time

def get_face_boxes(frame, model):
    """Process a frame and return face detection boxes for further processing."""
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

def logic(box):
    x1, y1, x2, y2 = box['coords']
    
    
    
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
        # Allow time for the Arduino to reset after serial connection
        time.sleep(2)
        # Clear any data in the buffer
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
        # Validate angle range (assuming 10-170 is valid range from serial-test.py)
        if 10 <= angle <= 170:
            # Send the angle value to Arduino
            ser.write(f"{angle}\n".encode())
            time.sleep(0.05)  # Wait for Arduino to process
            return True
        else:
            print(f"Invalid angle value: {angle}. Must be between 10 and 170.")
            return False
    except Exception as e:
        print(f"Error sending command: {e}")
        return False

# Keep model initialization as is
model = YOLO("yolov11s-face.pt")   

# Initialize serial connection for servo control
servo_connection = setup_servo_connection()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()   
    if not ret:
        break

    # Get face boxes for further processing
    face_boxes = get_face_boxes(frame, model)
    
    # Draw boxes on the frame
    frame = draw_boxes(frame, face_boxes)
    
   # angle = logic(face_boxes)
     
   # send_servo_command(servo_connection, angle)
     
    cv2.imshow("YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Close the serial connection if it's open
if 'servo_connection' in locals() and servo_connection and servo_connection.is_open:
    servo_connection.close()
    print("Serial connection closed.")
