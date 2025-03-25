/*
#include <Servo.h>

Servo myservo;  // create Servo object to control a servo
int val = 90;   // default position (middle)
int newVal;     // new value from serial

void setup() {
  Serial.begin(9600);       // initialize serial communication
  myservo.attach(9);        // attaches the servo on pin 9 to the Servo object
  myservo.write(val);       // set initial position
}

void loop() {
  if (Serial.available() > 0) {
    newVal = Serial.parseInt();       // read the value from the serial input
    if (newVal >= 10 && newVal <= 170) {
      val = newVal;                   // only update val if in valid range
    }
  }
  
  myservo.write(val);                 // constantly write the current position value
  delay(15);                          // small delay for stability
}
*/

#include <Servo.h>

Servo myservo;                // Create Servo object to control a servo

// PID Variables
double setpoint = 90;         // Desired servo position (degrees)
double input = 90;            // Current position
double output = 90;           // Output to servo
double error = 0;             // Error between setpoint and input
double lastError = 0;         // Previous error for derivative calculation
double cumError = 0;          // Cumulative error for integral calculation
double rateError = 0;         // Rate of change of error for derivative calculation

// PID Constants (tune these values as needed)
double kp = 1.0;              // Proportional gain
double ki = 0.1;              // Integral gain
double kd = 0.05;             // Derivative gain

// Time variables for PID calculation
unsigned long currentTime, previousTime;
double elapsedTime;
double lastPosition = 90;     // Store last position for simulation

void setup() {
  Serial.begin(9600);         // Initialize serial communication
  myservo.attach(9);          // Attaches the servo on pin 9 to the Servo object
  myservo.write(90);          // Set initial position to middle
  
  previousTime = millis();    // Initialize timing
}

void loop() {
  // Read new setpoint from serial if available
  if (Serial.available() > 0) {
    int newSetpoint = Serial.parseInt();       // Read the value from serial input
    if (newSetpoint >= 10 && newSetpoint <= 170) {
      setpoint = newSetpoint;                  // Update setpoint if in valid range
      Serial.print("New setpoint: ");
      Serial.println(setpoint);
    }
  }
  
  // Simulate actual servo position (in a real system you might use a potentiometer or encoder)
  // This simple simulation assumes the servo moves toward the setpoint with some lag
  input = lastPosition;
  lastPosition = input;
  
  // Compute PID at regular intervals
  currentTime = millis();
  elapsedTime = (currentTime - previousTime) / 1000.0; // Time in seconds
  
  // Compute PID output
  computePID(elapsedTime);
  
  // Constrain output to valid servo range
  output = constrain(output, 10, 170);
  
  // Send output to servo
  myservo.write(output);
  
  // Print values for debugging
  Serial.print("SP:");
  Serial.print(setpoint);
  Serial.print(" PV:");
  Serial.print(input);
  Serial.print(" OUT:");
  Serial.println(output);
  
  previousTime = currentTime;
  delay(50);  // Small delay for stability
}

void computePID(double deltaTime) {
  // Calculate error
  error = setpoint - input;
  
  // Proportional term
  double pTerm = kp * error;
  
  // Integral term with anti-windup
  cumError += error * deltaTime;
  cumError = constrain(cumError, -30, 30);  // Limit integral windup
  double iTerm = ki * cumError;
  
  // Derivative term
  rateError = (error - lastError) / deltaTime;
  double dTerm = kd * rateError;
  
  // Calculate total output
  output = pTerm + iTerm + dTerm + 90;  // Add 90 as the center position
  
  // Save error for next iteration
  lastError = error;
}
