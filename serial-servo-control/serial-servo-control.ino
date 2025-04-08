#include <Servo.h>

Servo panServo;  // Servo object for pan (horizontal) movement
Servo tiltServo; // Servo object for tilt (vertical) movement

int currentPanPos = 90;  // Current position of the pan servo
int currentTiltPos = 90; // Current position of the tilt servo
int targetPanPos = 90;   // Target position for pan servo
int targetTiltPos = 90;  // Target position for tilt servo

unsigned long lastMoveTime;      // Time of last position update
unsigned long lastCommandTime;   // Time of last received command
const int MOVE_THRESHOLD = 3;    // Only move if difference is greater than this value
const int SERVO_MIN = 10;        // Minimum servo angle
const int SERVO_MAX = 170;       // Maximum servo angle
const int COMMAND_TIMEOUT = 100; // Timeout for receiving complete commands (ms)

// Motion control parameters
const int UPDATE_INTERVAL = 20;     // Time between position updates (milliseconds)
const float MAX_SPEED = 6.0;        // Maximum speed in degrees per update interval
float Kp = 0.25;                    // Proportional gain - reduced to prevent oscillation
const float MIN_ANGLE_CHANGE = 4.0; // Minimum change in angle to process

// Buffer for incoming data
String inputBuffer = "";
boolean commandComplete = false;

void setup()
{
  Serial.begin(9600);   // Initialize serial communication
  panServo.attach(9);   // Attaches the pan servo on pin 9
  tiltServo.attach(6); // Attaches the tilt servo on pin 10

  panServo.write(currentPanPos); // Set initial positions
  tiltServo.write(currentTiltPos);

  lastMoveTime = millis(); // Initialize timing
  lastCommandTime = millis();

  // Clear any startup jitter
  delay(500);
}

void loop()
{
  // Process serial input
  while (Serial.available() > 0)
  {
    char inChar = (char)Serial.read();

    // If it's a digit or comma, add to buffer
    if (isDigit(inChar) || inChar == ',')
    {
      inputBuffer += inChar;
      lastCommandTime = millis();
    }
    // If it's a newline or carriage return, process the command
    else if (inChar == '\n' || inChar == '\r')
    {
      commandComplete = true;
    }
  }

  // If command is complete or we've timed out with data in the buffer
  if (commandComplete ||
      (inputBuffer.length() > 0 && millis() - lastCommandTime > COMMAND_TIMEOUT))
  {
    if (inputBuffer.length() > 0)
    {
      // Parse two comma-separated values for pan and tilt
      int commaIndex = inputBuffer.indexOf(',');

      if (commaIndex > 0)
      {
        // Two values provided
        String panAngleStr = inputBuffer.substring(0, commaIndex);
        String tiltAngleStr = inputBuffer.substring(commaIndex + 1);

        int newPanTarget = panAngleStr.toInt();
        int newTiltTarget = tiltAngleStr.toInt();

        // Constrain within safe limits
        newPanTarget = constrain(newPanTarget, SERVO_MIN, SERVO_MAX);
        newTiltTarget = constrain(newTiltTarget, SERVO_MIN, SERVO_MAX);

        // Only update if the changes are significant
        if (abs(newPanTarget - targetPanPos) >= MIN_ANGLE_CHANGE)
        {
          targetPanPos = newPanTarget;
        }

        if (abs(newTiltTarget - targetTiltPos) >= MIN_ANGLE_CHANGE)
        {
          targetTiltPos = newTiltTarget;
        }
      }
      else
      {
        // Fallback to single value (pan only) for backward compatibility
        int newTarget = inputBuffer.toInt();
        newTarget = constrain(newTarget, SERVO_MIN, SERVO_MAX);

        if (abs(newTarget - targetPanPos) >= MIN_ANGLE_CHANGE)
        {
          targetPanPos = newTarget;
        }
      }
    }

    // Clear the buffer and reset command status
    inputBuffer = "";
    commandComplete = false;
  }

  // Update servo positions at regular intervals
  unsigned long currentTime = millis();
  if (currentTime - lastMoveTime >= UPDATE_INTERVAL)
  {
    lastMoveTime = currentTime;

    // Update pan servo
    updateServoPosition(panServo, targetPanPos, currentPanPos);

    // Update tilt servo
    updateServoPosition(tiltServo, targetTiltPos, currentTiltPos);
  }
}

void updateServoPosition(Servo &servo, int &targetPos, int &currentPos)
{
  // Calculate position error (distance to target)
  int posError = targetPos - currentPos;

  if (abs(posError) > 0)
  {
    // Calculate speed - proportional to distance but limited to MAX_SPEED
    float speed = posError * Kp;

    // Limit the speed to the maximum allowed speed
    speed = constrain(speed, -MAX_SPEED, MAX_SPEED);

    // Update current position by the calculated speed
    if (abs(speed) < 0.1)
    {
      // Prevent very small movements that cause jitter
      currentPos = targetPos;
    }
    else
    {
      currentPos += speed;
    }

    // Ensure we stay within valid servo range
    currentPos = constrain(currentPos, SERVO_MIN, SERVO_MAX);

    // Move the servo to the updated position
    servo.write(round(currentPos));
  }
}
