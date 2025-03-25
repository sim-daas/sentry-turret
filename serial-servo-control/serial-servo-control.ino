#include <Servo.h>

Servo myservo;  // create Servo object to control a servo
int val;    // variable to read the value from the serial input

void setup() {
  Serial.begin(9600);  // initialize serial communication
  myservo.attach(9);  // attaches the servo on pin 9 to the Servo object
}

void loop() {
  if (Serial.available() > 0) {
    val = Serial.parseInt();            // read the value from the serial input
    val = constrain(val, 10, 170);       // constrain the value to be between 0 and 180
    myservo.write(val);                 // sets the servo position according to the input value
    delay(15);                          // waits for the servo to get there
  }
}