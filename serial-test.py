import serial
import time

def main():
    # Configure the serial connection (make sure baud rate matches your Arduino sketch)
    try:
        # Open serial port - '/dev/ttyACM0' is the standard port for Arduino on Linux
        ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
        print("Connected to Arduino on /dev/ttyACM0")

        # Allow time for the Arduino to reset after serial connection
        time.sleep(2)

        # Clear any data in the buffer
        ser.reset_input_buffer()

        # Print instructions
        print("\nTest multiple servo modes:")
        print("1. Enter a single value (e.g. '90') to control just the pan servo")
        print("2. Enter two comma-separated values (e.g. '90,45') to control both pan and tilt servos")
        print("Enter 'q' to quit at any time\n")

        while True:
            # Ask user for angle values
            try:
                input_text = input("Enter angle(s) (10-170 degrees) [pan] or [pan,tilt] or 'q' to quit: ")

                # Check if user wants to quit
                if input_text.lower() == 'q':
                    break

                # Check if input contains a comma (dual servo mode)
                if ',' in input_text:
                    # Split the input and parse both values
                    pan_str, tilt_str = input_text.split(',', 1)
                    pan_angle = int(pan_str.strip())
                    tilt_angle = int(tilt_str.strip())
                    
                    # Validate both angles
                    if 10 <= pan_angle <= 170 and 10 <= tilt_angle <= 170:
                        # Format command for both servos
                        command = f"{pan_angle},{tilt_angle}\n"
                        print(f"Sending: Pan={pan_angle}°, Tilt={tilt_angle}°")
                        ser.write(command.encode())
                    else:
                        print("Invalid value! Both angles must be between 10 and 170 degrees.")
                else:
                    # Single servo mode (backward compatibility)
                    pan_angle = int(input_text)
                    if 10 <= pan_angle <= 170:
                        # Send just the pan angle (Arduino will use default for tilt)
                        command = f"{pan_angle}\n"
                        print(f"Sending: Pan={pan_angle}° (single servo mode)")
                        ser.write(command.encode())
                    else:
                        print("Invalid value! Please enter a number between 10 and 170.")

                # Wait for Arduino to process
                time.sleep(0.1)

            except ValueError:
                print("Please enter valid numbers in the format: [pan] or [pan,tilt]")
            except KeyboardInterrupt:
                print("\nExiting...")
                break

    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        print("Make sure Arduino is connected and the port is correct.")
        print("You might need to run this script with sudo privileges.")

    finally:
        # Close the serial port if it's open
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial connection closed.")

if __name__ == "__main__":
    main()
