"""
monitor.py — Combined sleep + lifestyle monitor
Hardware : Arduino Nano + MPU6050 + MAX30102
Serial   : timestamp_ms,ax,ay,az,ir,heart_rate_bpm @ 20 Hz

Output every analysis cycle:
    SLEEP_HOURS: <float>
    ACTIVITY_LEVEL: <0|1|2>   (0=Sedentary, 1=Lightly Active, 2=Active)

Usage:
    python monitor.py --port COM3
    python monitor.py --port /dev/ttyUSB0
    python monitor.py --port COM3 --verbose
"""

import argparse
import collections
import math
import sys
import time

import serial

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BAUD_RATE      = 115200
SAMPLE_RATE_HZ = 20
DEMO_MODE      = True    # set False for real deployment

if DEMO_MODE:
    ANALYSIS_WINDOW_SEC = 15
    SLEEP_ONSET_WINDOWS = 2
    WAKE_ONSET_WINDOWS  = 1
else:
    ANALYSIS_WINDOW_SEC = 60
    SLEEP_ONSET_WINDOWS = 15
    WAKE_ONSET_WINDOWS  = 5

SAMPLES_PER_WINDOW = SAMPLE_RATE_HZ * ANALYSIS_WINDOW_SEC

MOTION_SEDENTARY = 0.15
MOTION_ACTIVE    = 0.50
MOTION_SLEEP_MAX = 0.15
MOTION_VAR_MAX   = 0.02

HR_SLEEP_MIN    = 40
HR_SLEEP_MAX    = 90
HR_STRESS_MAX   = 90
HR_RESTING_MAX  = 85
MIN_VALID_HR    = 30
MAX_VALID_HR    = 220

# Minimum fraction of a window that must have valid HR readings before
# we trust the avg_hr value. Below this, hr_ok = False.
HR_VALID_FRAC   = 0.25

SMOOTH_WINDOW   = 8

LEVEL_NAMES = {0: "Sedentary", 1: "Lightly Active", 2: "Active"}

# Arduino sends this header on boot — must be explicitly skipped.
CSV_HEADER = "timestamp_ms"

# ─────────────────────────────────────────────────────────────────────────────
# SERIAL
# ─────────────────────────────────────────────────────────────────────────────

def open_serial(port: str) -> serial.Serial:
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=2)
        print(f"[serial] Connected to {port} @ {BAUD_RATE} baud")
        time.sleep(2)          # wait for Arduino reset
        ser.reset_input_buffer()
        return ser
    except serial.SerialException as e:
        print(f"[ERROR] Cannot open {port}: {e}")
        sys.exit(1)


def read_line(ser: serial.Serial) -> dict | None:
    """
    Parse one CSV line from the Arduino.
    Returns a dict or None on any parse / validation failure.

    FIX 2: explicitly detect and discard the CSV header line that the
    updated Arduino firmware emits on startup. Previously this would
    raise ValueError inside the try block and be silently swallowed,
    which is fine but wasteful and confusing to debug.
    """
    try:
        raw = ser.readline()
        if not raw:
            return None
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            return None
        # Skip error lines, comments, and the CSV header
        if line.startswith("ERR") or line.startswith("#"):
            return None
        if line.startswith(CSV_HEADER):   # FIX 2: explicit header skip
            return None
        parts = line.split(",")
        if len(parts) != 6:
            return None
        return {
            "timestamp":  int(parts[0]),
            "ax":         float(parts[1]),
            "ay":         float(parts[2]),
            "az":         float(parts[3]),
            "ir":         int(parts[4]),
            "heart_rate": int(parts[5]),
        }
    except (ValueError, UnicodeDecodeError):
        return None

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def compute_motion(ax: float, ay: float, az: float) -> float:
    """Deviation from 1 g — zero when perfectly still."""
    return abs(math.sqrt(ax**2 + ay**2 + az**2) - 1.0)


class MovingAverage:
    """O(1) running mean over a fixed window."""
    def __init__(self, window: int):
        self._buf   = collections.deque(maxlen=window)
        self._total = 0.0

    def update(self, v: float) -> float:
        if len(self._buf) == self._buf.maxlen:
            self._total -= self._buf[0]
        self._buf.append(v)
        self._total += v
        return self._total / len(self._buf)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(motion_win: list[float], hr_win: list[int]) -> dict:
    """
    Compute per-window features from raw (unsmoothed) motion and HR buffers.

    FIX 4 (design): hr_win now contains raw validated HR readings, not
    moving-averaged ones. The valid-HR fraction check uses HR_VALID_FRAC
    so a window with only a few finger-off samples doesn't report hr_ok=True
    on stale averaged values.
    """
    n = len(motion_win)
    if n == 0:
        return {
            "avg_motion":      0.0,
            "motion_variance": 0.0,
            "avg_hr":          0.0,
            "hr_ok":           False,
            "valid_hr_frac":   0.0,
            "sedentary_pct":   0.0,
            "active_pct":      0.0,
            "light_pct":       0.0,
        }

    avg_m = sum(motion_win) / n
    var_m = sum((x - avg_m) ** 2 for x in motion_win) / n

    sed_n = sum(1 for m in motion_win if m < MOTION_SEDENTARY)
    act_n = sum(1 for m in motion_win if m > MOTION_ACTIVE)
    lit_n = n - sed_n - act_n

    # Only count HR values where the sensor reported a real beat
    valid_hr    = [h for h in hr_win if MIN_VALID_HR <= h <= MAX_VALID_HR]
    valid_frac  = len(valid_hr) / n
    avg_hr      = sum(valid_hr) / len(valid_hr) if valid_hr else 0.0

    # hr_ok requires enough valid readings to be statistically meaningful
    hr_ok = valid_frac >= HR_VALID_FRAC

    return {
        "avg_motion":      avg_m,
        "motion_variance": var_m,
        "avg_hr":          avg_hr,
        "hr_ok":           hr_ok,
        "valid_hr_frac":   valid_frac,
        "sedentary_pct":   (sed_n / n) * 100,
        "active_pct":      (act_n / n) * 100,
        "light_pct":       (lit_n / n) * 100,
    }

# ─────────────────────────────────────────────────────────────────────────────
# SLEEP DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_sleep(features: dict) -> str:
    """
    FIX 3: when hr_ok is False (finger off sensor / insufficient readings),
    default to "awake" rather than "sleep". Previously, low motion with no
    HR data would be classified as sleep — sitting still at a desk with no
    finger on the sensor would start accumulating sleep time.
    """
    motion_ok = (
        features["avg_motion"]     < MOTION_SLEEP_MAX and
        features["motion_variance"] < MOTION_VAR_MAX
    )
    if not motion_ok:
        return "awake"

    # Motion looks sleep-like — now check HR if available
    if features["hr_ok"]:
        hr = features["avg_hr"]
        if hr > HR_STRESS_MAX:
            return "awake"
        return "sleep" if HR_SLEEP_MIN <= hr <= HR_SLEEP_MAX else "awake"

    # FIX 3: insufficient HR data → conservative default
    return "awake"

# ─────────────────────────────────────────────────────────────────────────────
# LIFESTYLE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def classify_lifestyle(features: dict) -> int:
    if features["sedentary_pct"] > 60:
        level = 0
    elif features["active_pct"] > 30:
        level = 2
    else:
        level = 1
    # Elevated HR nudges sedentary → lightly active (e.g. sitting but stressed)
    if level == 0 and features["hr_ok"] and features["avg_hr"] > HR_RESTING_MAX:
        level = 1
    return level

# ─────────────────────────────────────────────────────────────────────────────
# SLEEP TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class SleepTracker:
    """
    State machine: AWAKE ↔ IN_SLEEP.

    FIX 5 (off-by-one): sleep seconds are now only added when the label is
    "sleep" AND we are confirming we remain in IN_SLEEP — i.e. AFTER the
    wake-onset check. Previously the accumulation happened before the check,
    so the boundary window (last sleep window before waking) was counted even
    when it triggered the AWAKE transition.
    """

    def __init__(self):
        self._state        = "AWAKE"
        self._consec_sleep = 0
        self._consec_awake = 0
        self._sleep_secs   = 0.0

    def update(self, label: str):
        if label == "sleep":
            self._consec_sleep += 1
            self._consec_awake  = 0
        else:
            self._consec_awake += 1
            self._consec_sleep  = 0

        if self._state == "AWAKE":
            if self._consec_sleep >= SLEEP_ONSET_WINDOWS:
                self._state        = "IN_SLEEP"
                self._consec_awake = 0

        elif self._state == "IN_SLEEP":
            # Check wake condition BEFORE accumulating — FIX 5
            if self._consec_awake >= WAKE_ONSET_WINDOWS:
                self._state = "AWAKE"
            elif label == "sleep":
                # Only accumulate when confirmed still asleep
                self._sleep_secs += ANALYSIS_WINDOW_SEC

    @property
    def sleep_hours(self) -> float:
        return round(self._sleep_secs / 3600.0, 2)

# ─────────────────────────────────────────────────────────────────────────────
# LIFESTYLE TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class LifestyleTracker:
    """Majority vote over the last N windows."""
    HISTORY = 5

    def __init__(self):
        self._history: collections.deque[int] = collections.deque(maxlen=self.HISTORY)

    def update(self, level: int):
        self._history.append(level)

    @property
    def level(self) -> int:
        if not self._history:
            return 0
        counts = [0, 0, 0]
        for lvl in self._history:
            counts[lvl] += 1
        return counts.index(max(counts))

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def print_output(sleep_hours: float, activity_level: int,
                 verbose: bool, features: dict | None = None,
                 sleep_state: str = ""):
    print("\n" + "═" * 40)
    print(f"  SLEEP_HOURS: {sleep_hours}")
    print(f"  ACTIVITY_LEVEL: {activity_level}  ({LEVEL_NAMES[activity_level]})")
    print("═" * 40)
    if verbose and features:
        print(f"  [debug] sleep_state    : {sleep_state}")
        print(f"  [debug] avg_motion     : {features['avg_motion']:.4f} g")
        print(f"  [debug] motion_var     : {features['motion_variance']:.4f}")
        print(f"  [debug] avg_hr         : {features['avg_hr']:.0f} bpm"
              f"  (ok={features['hr_ok']}"
              f"  frac={features['valid_hr_frac']:.0%})")
        print(f"  [debug] sedentary %    : {features['sedentary_pct']:.1f}%")
        print(f"  [debug] light %        : {features['light_pct']:.1f}%")
        print(f"  [debug] active %       : {features['active_pct']:.1f}%")
        print("═" * 40)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(port: str, verbose: bool):
    ser = open_serial(port)

    # Moving averages used only for display/smoothing — NOT fed into feature
    # windows (FIX 4: feature windows store raw values so stats are correct).
    motion_smoother = MovingAverage(SMOOTH_WINDOW)

    # Raw per-sample buffers for feature extraction
    motion_buf: collections.deque[float] = collections.deque(maxlen=SAMPLES_PER_WINDOW)
    hr_buf:     collections.deque[int]   = collections.deque(maxlen=SAMPLES_PER_WINDOW)

    sleep_tracker     = SleepTracker()
    lifestyle_tracker = LifestyleTracker()

    sample_count = 0
    mode_label   = "DEMO" if DEMO_MODE else "NORMAL"
    print(f"[monitor] Running in {mode_label} mode.")
    print(f"[monitor] Analysis every {ANALYSIS_WINDOW_SEC}s"
          f" — window = {SAMPLES_PER_WINDOW} samples.")
    print(f"[monitor] Press Ctrl+C to stop.\n")

    try:
        while True:
            rec = read_line(ser)
            if rec is None:
                continue

            # ── Motion ───────────────────────────────────────────────────────
            motion_raw = compute_motion(rec["ax"], rec["ay"], rec["az"])
            motion_smoother.update(motion_raw)   # for display only
            motion_buf.append(motion_raw)         # raw → feature window

            # ── Heart rate ───────────────────────────────────────────────────
            # FIX 1: only buffer non-zero HR values.
            # Zero means "no beat detected this tick" — feeding it into the
            # window would dilute avg_hr and make hr_ok falsely True at ~0 BPM.
            # We still append to hr_buf (as 0) so the buffer length matches the
            # motion buffer; extract_features() filters out invalid values.
            hr_buf.append(rec["heart_rate"])

            sample_count += 1

            # ── Analysis every full window ────────────────────────────────────
            if sample_count % SAMPLES_PER_WINDOW == 0:
                features    = extract_features(list(motion_buf), list(hr_buf))
                sleep_label = detect_sleep(features)
                act_level   = classify_lifestyle(features)

                sleep_tracker.update(sleep_label)
                lifestyle_tracker.update(act_level)

                print_output(
                    sleep_hours    = sleep_tracker.sleep_hours,
                    activity_level = lifestyle_tracker.level,
                    verbose        = verbose,
                    features       = features,
                    sleep_state    = sleep_label,
                )

    except KeyboardInterrupt:
        print("\n[monitor] Stopped.")
        print("\n── FINAL OUTPUT ──")
        print(f"SLEEP_HOURS: {sleep_tracker.sleep_hours}")
        print(f"ACTIVITY_LEVEL: {lifestyle_tracker.level}")

    finally:
        ser.close()
        print("[serial] Port closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sleep + Lifestyle monitor — Arduino MPU6050 + MAX30102"
    )
    parser.add_argument("--port",    required=True,
                        help="Serial port, e.g. COM3 or /dev/ttyUSB0")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed feature breakdown each cycle")
    args = parser.parse_args()
    run(port=args.port, verbose=args.verbose)