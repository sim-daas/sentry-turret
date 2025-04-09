import serial
import time

def main():
    try:
        ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
        print("Connected to Arduino on /dev/ttyACM0")

        time.sleep(2)

        ser.reset_input_buffer()

        print("\nTest multiple servo modes:")
        print("1. Enter a single value (e.g. '90') to control just the pan servo")
        print("2. Enter two comma-separated values (e.g. '90,45') to control both pan and tilt servos")
        print("Enter 'q' to quit at any time\n")

        while True:
            try:
                inp = input("Enter angle(s) (10-170 degrees) [pan] or [pan,tilt] or 'q' to quit: ")

                if inp.lower() == 'q':
                    break

                if ',' in inp:
                    pan_str, tlt_str = inp.split(',', 1)
                    p_ang = int(pan_str.strip())
                    t_ang = int(tlt_str.strip())
                    
                    if 10 <= p_ang <= 170 and 10 <= t_ang <= 170:
                        cmd = f"{p_ang},{t_ang}\n"
                        print(f"Sending: Pan={p_ang}°, Tilt={t_ang}°")
                        ser.write(cmd.encode())
                    else:
                        print("Invalid value! Both angles must be between 10 and 170 degrees.")
                else:
                    p_ang = int(inp)
                    if 10 <= p_ang <= 170:
                        cmd = f"{p_ang}\n"
                        print(f"Sending: Pan={p_ang}° (single servo mode)")
                        ser.write(cmd.encode())
                    else:
                        print("Invalid value! Please enter a number between 10 and 170.")

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
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial connection closed.")

if __name__ == "__main__":
    main()
