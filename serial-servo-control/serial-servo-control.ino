#include <Servo.h>

Servo myservo;                   // Create Servo object to control a servo
int currentPos = 90;             // Current position of the servo (starting at middle)
int targetPos = 90;              // Target position (where we want to go)
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
const float MIN_ANGLE_CHANGE = 4.0; // Minimum change in angle to process (transferred from Python)

// Buffer for incoming data
String inputBuffer = "";
boolean commandComplete = false;

void setup()
{
  Serial.begin(9600);        // Initialize serial communication
  myservo.attach(9);         // Attaches the servo on pin 9 to the Servo object
  myservo.write(currentPos); // Set initial position
  lastMoveTime = millis();   // Initialize the last move time
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

    // If it's a digit, add to buffer
    if (isDigit(inChar))
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
      int newTarget = inputBuffer.toInt();

      // Constrain the target within safe limits
      newTarget = constrain(newTarget, SERVO_MIN, SERVO_MAX);

      // Only update if the change is significant (enhanced threshold check)
      if (abs(newTarget - targetPos) >= MIN_ANGLE_CHANGE)
      {
        targetPos = newTarget;
      }
    }

    // Clear the buffer and reset command status
    inputBuffer = "";
    commandComplete = false;
  }

  // Update servo position at regular intervals
  unsigned long currentTime = millis();
  if (currentTime - lastMoveTime >= UPDATE_INTERVAL)
  {
    lastMoveTime = currentTime;

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
      myservo.write(round(currentPos));
    }
  }
}
