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
