# AI Sentry Gun Project

## Overview

This project creates an intelligent, autonomous sentry gun system that uses computer vision and machine learning to detect, track, and aim at faces or objects in real-time. The system calculates the position of detected targets and sends precise angle commands to a servo-based pan/tilt mechanism, allowing it to aim a laser pointer, nerf gun, or airsoft gun at the target with high accuracy.

## Features

- Real-time face and object detection using YOLOv11
- Object tracking across video frames
- Interactive object selection interface
- Smooth servo motion control
- FPS monitoring and performance statistics
- Horizontal image flipping option
- Support for various camera types
- Arduino-based servo control

## Hardware Requirements

### Processing Unit (one of the following)
- Laptop/Desktop with CUDA-capable GPU (recommended)
- Jetson Nano / Jetson Xavier NX
- Raspberry Pi 4 (with reduced performance expectations)

### Microcontroller
- Arduino Uno/Nano/Mega

### Camera (one of the following)
- USB webcam (720p or higher resolution)
- CSI camera (for Jetson Nano)
- Built-in laptop camera

### Servo System
- Standard servo motor for pan (horizontal) movement
- Optional second servo for tilt (vertical) movement
- Pan/tilt bracket (can be 3D printed or purchased)

### Targeting Device (one of the following)
- Laser pointer module
- Nerf gun
- Airsoft gun (low power)
- LED indicator

### Other Components
- Jumper wires
- 5V power supply (for servos)
- USB cable (Type A to B for Arduino)
- Mounting hardware

## Software Architecture

The system consists of three primary components:

1. **Object Detection and Tracking (Python)**: Uses YOLOv11 to detect faces/objects and track them across frames
2. **Position Calculation (Python)**: Converts object position to servo angles
3. **Servo Control (Arduino C++)**: Receives angle commands and smoothly controls servo motors

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/face-track.git
cd face-track
```

### 2. Install Python Dependencies

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Download YOLO Models

Download the YOLOv11 face detection model:

```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov11n-face.pt
```

Alternative models (depending on your needs):
- `yolov11n.pt` - General object detection (Nano version, fastest)
- `yolov11s.pt` - General object detection (Small version)
- `yolov11m.pt` - General object detection (Medium version)
- `yolov11l.pt` - General object detection (Large version, most accurate)

### 4. Arduino Setup

1. Install the Arduino IDE from [arduino.cc](https://www.arduino.cc/en/software)
2. Open the `serial-servo-control/serial-servo-control.ino` file in the Arduino IDE
3. Connect your Arduino board to your computer
4. Select the correct board and port in the Arduino IDE
5. Upload the sketch to your Arduino

## Hardware Setup

### Servo Connections

1. Connect the servo signal wire to Arduino pin 9 (pan servo)
2. Connect servo power (red) to a 5V power supply
3. Connect servo ground (black/brown) to Arduino ground and power supply ground

![Servo Wiring Diagram](https://via.placeholder.com/400x300?text=Servo+Wiring+Diagram)

### Camera Setup

- For USB webcams: Simply connect to an available USB port
- For CSI cameras on Jetson: Connect to the CSI port as per Jetson documentation
- For built-in cameras: No additional setup needed

### Mechanical Assembly

1. Mount the servo motor securely to a stable base
2. Attach the targeting device (laser, nerf gun, etc.) to the servo arm
3. Position the camera near the targeting device for better aim alignment

## File Descriptions

### Python Files

- **`facedet.py`**: The main application that handles:
  - Camera image capture and processing
  - Face/object detection using YOLOv11
  - Tracking objects across video frames
  - Object selection interface
  - Position-to-angle calculations
  - Servo control commands via serial communication
  - Performance monitoring (FPS counter)

- **`serial-test.py`**: A utility script to test communication with the Arduino:
  - Allows sending manual angle commands through the serial port
  - Useful for testing servo movement and calibration
  - Helps diagnose connection issues between computer and Arduino

### Arduino Files

- **`serial-servo-control.ino`**: The Arduino sketch that:
  - Receives angle commands via serial port
  - Implements smooth movement control using proportional control
  - Prevents jitter and sudden movements
  - Ensures servo stays within safe movement limits
  - Provides command validation and error handling

## Usage Instructions

### Starting the System

1. Ensure the Arduino is connected and has the sketch uploaded
2. Connect the camera and make sure it's recognized
3. Run the main script:

```bash
python facedet.py
```

### First-Time Setup

1. When the script starts, it will display the first camera frame
2. You'll be prompted to select a target object to track:
   - Use arrow keys to navigate between detected objects
   - Press SPACE to confirm your selection
   - Press ESC to automatically track the first detected object

### Keyboard Controls

During operation, you can use these keyboard controls:

- **Q**: Quit the application
- **T**: Toggle tracking on/off
- **R**: Reset selected object (pick a new target)
- **F**: Toggle image horizontal flip

### Serial Testing

To test the servo connection independently of the detection system:

```bash
python serial-test.py
```

Follow the prompts to send angle commands directly to the servo.

## Customization Options

### Camera Selection

To use a different camera, modify the camera index in `facedet.py`:

```python
# Change /dev/video2 to 0 for default camera, or another index for specific cameras
cap = cv2.VideoCapture('/dev/video2')
```

### Detection Models

Change the YOLO model to detect different objects:

```python
# For general object detection instead of just faces
model = YOLO("yolov11n.pt")
```

### Tracking Sensitivity

Adjust the smoothing factor for more or less responsive tracking:

```python
# Lower values (0.1-0.3) for smoother tracking, higher values (0.4-0.9) for more responsive tracking
smoothing_factor = 0.3
```

### Servo Movement

Modify servo speed and responsiveness in the Arduino sketch:

```cpp
// Higher values for faster movement, lower for slower, more precise movement
const float MAX_SPEED = 6.0;

// Higher values for more responsive but potentially jerky movement
float Kp = 0.25;
```

## Troubleshooting

### Common Issues

1. **Camera not found**: Check camera connection and update the device index in `facedet.py`
2. **Arduino connection error**: Verify the correct port in `setup_servo_connection()` function
3. **Servo not moving**: Check wiring and power supply for the servo
4. **Detection not working**: Ensure the YOLO model file is in the correct location
5. **Jerky servo movements**: Decrease `MAX_SPEED` and/or `Kp` in the Arduino sketch

### Performance Optimization

- For slower computers, use smaller models like `yolov11n-face.pt`
- Increase the `command_interval` in `facedet.py` to reduce CPU usage
- Reduce the camera resolution if needed

## Safety Considerations

- This project is designed for educational purposes
- If using with airsoft or nerf guns, always follow safety guidelines:
  - Never aim at people or animals
  - Use eye protection
  - Follow local regulations regarding projectile devices

## Future Enhancements

- Add vertical (tilt) servo support
- Implement camera-based distance estimation
- Add motion prediction algorithms
- Create a web interface for remote control
- Integrate with additional sensors (PIR, ultrasonic, etc.)

## License

This project is released under the MIT License. See the LICENSE file for details.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv11 implementation
- OpenCV team for computer vision libraries
- Arduino community for servo control examples

## Contact

If you have questions or want to share your build, please create an issue in the GitHub repository.