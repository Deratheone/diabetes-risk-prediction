import argparse
import time
import serial

parser = argparse.ArgumentParser()
parser.add_argument("--port", required=True)
args = parser.parse_args()

ser = serial.Serial(args.port, 115200, timeout=2)
print(f"Connected to {args.port}\n")
time.sleep(2)
ser.reset_input_buffer()

for _ in range(30):
    raw = ser.readline()
    if raw:
        print(repr(raw))

ser.close()
