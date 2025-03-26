#include <Servo.h>

Servo myservo;              // Create Servo object to control a servo
int currentPos = 90;        // Current position of the servo (starting at middle)
int targetPos = 90;         // Target position (where we want to go)
unsigned long lastMoveTime; // Time of last position update

// Motion control parameters
const int UPDATE_INTERVAL = 15;    // Time between position updates (milliseconds)
const float MAX_SPEED = 2.0;       // Maximum speed in degrees per update interval
float Kp = 0.3;                    // Proportional gain - controls speed based on distance

void setup() {
  Serial.begin(9600);              // Initialize serial communication
  myservo.attach(9);               // Attaches the servo on pin 9 to the Servo object
  myservo.write(currentPos);       // Set initial position
  lastMoveTime = millis();         // Initialize the last move time
}

void loop() {
  // Check for new target position from serial
  if (Serial.available() > 0) {
    int newTarget = Serial.parseInt();  // Read the value from serial input
    targetPos = newTarget;
  }

  // Update servo position at regular intervals
  unsigned long currentTime = millis();
  if (currentTime - lastMoveTime >= UPDATE_INTERVAL) {
    lastMoveTime = currentTime;
    
    // Calculate position error (distance to target)
    int posError = targetPos - currentPos;
    
    if (abs(posError) > 0) {
      // Calculate speed - proportional to distance but limited to MAX_SPEED
      // This is the 'P' part of a PID controller
      float speed = posError * Kp;
      
      // Limit the speed to the maximum allowed speed
      if (speed > MAX_SPEED) speed = MAX_SPEED;
      if (speed < -MAX_SPEED) speed = -MAX_SPEED;
      
      // Update current position by the calculated speed
      currentPos += speed;
      
      // Ensure we don't overshoot the target by small amounts due to floating-point math
      if ((speed > 0 && currentPos > targetPos) || 
          (speed < 0 && currentPos < targetPos)) {
        currentPos = targetPos;
      }
      
      // Move the servo to the updated position
      myservo.write(currentPos);
    }
  }
}
