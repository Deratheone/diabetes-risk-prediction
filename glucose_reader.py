"""
glucose_reader.py — Arduino Urine Glucose Strip Analyzer Reader
Hardware: Arduino with TCS34725 RGB color sensor
Serial: 9600 baud (COM5)

Reads RGB color data from the glucose analyzer and matches it against
predefined glucose level reference colors.

Output format (JSON):
{
    "success": true/false,
    "error": error_message (if success=false),
    "glucose_level": "No glucose detected | Low (~100 mg/dL) | Medium (~250 mg/dL) | High (~500+ mg/dL)",
    "glucose_value": 100/250/500 (numeric estimate),
    "match_distance": 5.1 (Euclidean RGB distance),
    "confidence": "high/medium/low",
    "is_no_strip": false,
    "serial_output": [...excerpt of output lines...]
}
"""

import serial
import time
import re
from typing import dict, list

# Configuration
PORT = "COM5"
BAUD_RATE = 9600
TIMEOUT = 60  # Max wait time for Arduino
READ_TIMEOUT = 10  # Timeout for individual serial reads

# Glucose reference colors (RGB values from glucose_analysis.ino)
GLUCOSE_LEVELS = [
    {"label": "!! NO STRIP DETECTED !!", "r": 119, "g": 82, "b": 56, "is_no_strip": True},
    {"label": "Normal — No glucose detected", "r": 104, "g": 87, "b": 56, "value": 0, "is_no_strip": False},
    {"label": "Low glucose (~100 mg/dL)", "r": 108, "g": 86, "b": 55, "value": 100, "is_no_strip": False},
    {"label": "Medium glucose (~250 mg/dL)", "r": 114, "g": 82, "b": 50, "value": 250, "is_no_strip": False},
    {"label": "High glucose (~500+ mg/dL)", "r": 117, "g": 80, "b": 50, "value": 500, "is_no_strip": False},
]

LOW_CONFIDENCE_THRESHOLD = 20.0  # Distance threshold for low confidence


def connect_to_serial(port: str = PORT, baud: int = BAUD_RATE, timeout: int = 5) -> dict:
    """
    Attempt to connect to the Arduino device.
    
    Returns:
        {
            "success": bool,
            "message": str,
            "ser": serial.Serial or None
        }
    """
    try:
        ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2)  # Wait for Arduino to reset
        ser.reset_input_buffer()
        return {
            "success": True,
            "message": f"Connected to {port} @ {baud} baud",
            "ser": ser
        }
    except serial.SerialException as e:
        return {
            "success": False,
            "message": f"Cannot connect to {port}: {str(e)}",
            "ser": None
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Unexpected error: {str(e)}",
            "ser": None
        }


def wait_for_prompt(ser: serial.Serial, timeout: int = 30) -> dict:
    """
    Wait for the Arduino prompt asking to press ENTER.
    
    Returns:
        {
            "success": bool,
            "output": [...lines read...],
            "found_prompt": bool
        }
    """
    output_lines = []
    start_time = time.time()
    found_prompt = False
    
    try:
        while time.time() - start_time < timeout:
            if ser.in_waiting:
                raw = ser.readline()
                if raw:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if line:
                        output_lines.append(line)
                        # Look for the prompt asking to press ENTER
                        if "press ENTER" in line.lower() or "then press" in line.lower():
                            found_prompt = True
            else:
                time.sleep(0.1)
        
        return {
            "success": True,
            "output": output_lines,
            "found_prompt": found_prompt
        }
    except Exception as e:
        return {
            "success": False,
            "output": output_lines,
            "found_prompt": False,
            "error": str(e)
        }


def send_enter_and_read_result(ser: serial.Serial, timeout: int = 30) -> dict:
    """
    Send ENTER key to Arduino and read the complete result.
    
    Returns:
        {
            "success": bool,
            "output": [...all lines...],
            "glucose_level": str,
            "glucose_value": int or None,
            "match_distance": float or None,
            "confidence": str,
            "is_no_strip": bool,
            "error": str (if any)
        }
    """
    output_lines = []
    start_time = time.time()
    
    try:
        # Send ENTER
        ser.write(b'\r\n')
        time.sleep(0.5)
        
        # Read all output until we see the final RESULT section
        reading_result = False
        found_result_end = False
        match_distance = None
        glucose_level = None
        is_no_strip = False
        
        while time.time() - start_time < timeout:
            if ser.in_waiting:
                raw = ser.readline()
                if raw:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if line:
                        output_lines.append(line)
                        
                        # Detect result section
                        if "RESULT" in line and "========" in line:
                            reading_result = True
                        
                        if reading_result:
                            # Parse glucose level
                            if "Glucose Level" in line:
                                glucose_level = line.split(":", 1)[-1].strip()
                            
                            # Parse match distance
                            elif "Match distance" in line:
                                try:
                                    match_str = line.split(":", 1)[-1].strip()
                                    match_distance = float(match_str)
                                except:
                                    pass
                            
                            # Check for no strip
                            elif "!! NO STRIP DETECTED !!" in line:
                                is_no_strip = True
                            
                            # End of result section (reset indicator)
                            elif "Reset the Arduino" in line:
                                found_result_end = True
                        
                        # Check for completion
                        if found_result_end:
                            break
            else:
                time.sleep(0.1)
        
        # Determine confidence
        confidence = "unknown"
        if is_no_strip:
            confidence = "low"
        elif match_distance is not None:
            if match_distance > LOW_CONFIDENCE_THRESHOLD:
                confidence = "low"
            elif match_distance > 15:
                confidence = "medium"
            else:
                confidence = "high"
        
        # Find glucose value from matching
        glucose_value = None
        if glucose_level and not is_no_strip:
            for level in GLUCOSE_LEVELS:
                if level["label"] in glucose_level or glucose_level in level["label"]:
                    glucose_value = level.get("value", 0)
                    break
        
        return {
            "success": True,
            "output": output_lines,
            "glucose_level": glucose_level or "Unknown",
            "glucose_value": glucose_value,
            "match_distance": match_distance,
            "confidence": confidence,
            "is_no_strip": is_no_strip
        }
    
    except Exception as e:
        return {
            "success": False,
            "output": output_lines,
            "error": str(e)
        }


def read_glucose(port: str = PORT) -> dict:
    """
    Main function to read glucose data from Arduino.
    
    Returns:
        {
            "success": bool,
            "glucose_level": str,
            "glucose_value": int,
            "match_distance": float,
            "confidence": str,
            "is_no_strip": bool,
            "error": str (if any),
            "serial_output": [...excerpt...]
        }
    """
    # Step 1: Connect
    conn = connect_to_serial(port)
    if not conn["success"]:
        return {
            "success": False,
            "error": conn["message"]
        }
    
    ser = conn["ser"]
    
    try:
        # Step 2: Wait for prompt
        prompt_result = wait_for_prompt(ser)
        if not prompt_result["success"]:
            return {
                "success": False,
                "error": f"Failed to read from serial: {prompt_result.get('error', 'Unknown error')}"
            }
        
        if not prompt_result["found_prompt"]:
            return {
                "success": False,
                "error": "Arduino did not send the expected prompt. Is the glucose analyzer connected and running?"
            }
        
        # Step 3: Send ENTER and read result
        result = send_enter_and_read_result(ser)
        if not result["success"]:
            return {
                "success": False,
                "error": f"Failed to read result: {result.get('error', 'Unknown error')}"
            }
        
        # Return formatted result
        return {
            "success": True,
            "glucose_level": result["glucose_level"],
            "glucose_value": result["glucose_value"],
            "match_distance": result["match_distance"],
            "confidence": result["confidence"],
            "is_no_strip": result["is_no_strip"],
            "serial_output": result["output"][-10:] if result["output"] else []  # Last 10 lines
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }
    
    finally:
        try:
            ser.close()
        except:
            pass


if __name__ == "__main__":
    # Test the reader
    print("Testing Glucose Reader...")
    print("Ensure Arduino glucose analyzer is connected to COM5")
    print("=" * 60)
    
    result = read_glucose()
    
    import json
    print(json.dumps(result, indent=2))
