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

        while True:
            # Ask user for pulse width value
            try:
                pulse = input("Enter pulse width (500-2500 μs) or 'q' to quit: ")

                # Check if user wants to quit
                if pulse.lower() == 'q':
                    break

                # Convert input to integer and validate range
                pulse_value = int(pulse)
                if 10 <= pulse_value <= 170:
                    # Send the pulse width value to Arduino
                    ser.write(f"{pulse_value}\n".encode())

                    # Read and print the response from Arduino
                    time.sleep(0.1)  # Wait for Arduino to respond
                else:
                    print("Invalid value! Please enter a number between 500 and 2500.")

            except ValueError:
                print("Please enter a valid number.")
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
