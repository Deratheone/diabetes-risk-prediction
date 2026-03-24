/*
 * Combined Sensor Streamer — Arduino Nano
 * Hardware : Arduino Nano + MPU6050 + MAX30102
 * Output   : CSV @ 20 Hz  →  timestamp_ms,ax,ay,az,ir,heart_rate_bpm
 *
 * Bug fixes vs original:
 *   1. PPG FIFO is now drained on EVERY loop iteration (not just at output rate).
 *      checkForBeat() needs to see every sample at sensor rate — at 400 sps with
 *      a 50 ms output tick you were feeding it 1 in 20 samples. Fixed by using
 *      ppg.check() + ppg.getFIFOIR() + ppg.nextSample() inside the fast loop.
 *   2. Sensor downgraded to 100 sps (5 samples per 50 ms tick). Still plenty for
 *      BPM detection; prevents FIFO overflow and reduces Nano I2C load.
 *   3. First-beat skip: when lastBeatMs == 0, delta = millis()-0 = huge garbage.
 *      Now explicitly detected and used only to seed the timestamp.
 *   4. Outlier rejection added before buffering — a single motion artifact can no
 *      longer corrupt the running average.
 *   5. Removed conflicting setPulseAmplitudeRed/IR calls that overrode setup().
 *   6. lastIrValue cached globally so the output section always has a fresh reading
 *      even after the FIFO was drained mid-tick.
 */

#include <Wire.h>
#include <MPU6050.h>
#include "MAX30105.h"
#include "heartRate.h"

#define SERIAL_BAUD         115200
#define OUTPUT_INTERVAL_MS  50          // 20 Hz
#define ACCEL_SCALE         16384.0f

// Heart rate
#define RATE_BUF_SIZE       6
#define IR_FINGER_THRESH    50000L
#define MIN_BEAT_MS         333L        // 333 ms → 180 BPM ceiling
#define MAX_BEAT_MS         1500L       // 1500 ms → 40 BPM floor
#define OUTLIER_RATIO       0.35f       // reject if > 35% from running avg

MPU6050  mpu;
MAX30105 ppg;

// ── HR state (modified only in processPPGSample) ─────────────────────────────
float  rateBuf[RATE_BUF_SIZE];
byte   rateBufIdx  = 0;
bool   rateBufFull = false;
long   lastBeatMs  = 0;
int    beatAvg     = 0;

// Last IR value seen this tick (updated while draining FIFO)
volatile long lastIrValue = 0;

unsigned long prevOutputMs = 0;

// ── Process one PPG sample ───────────────────────────────────────────────────
// Called at the sensor's sample rate (100 sps), not the output rate (20 Hz).
void processPPGSample(long ir) {
  lastIrValue = ir;

  if (ir < IR_FINGER_THRESH) {
    // Finger removed — reset everything cleanly
    for (byte i = 0; i < RATE_BUF_SIZE; i++) rateBuf[i] = 0;
    rateBufIdx  = 0;
    rateBufFull = false;
    lastBeatMs  = 0;
    beatAvg     = 0;
    return;
  }

  if (!checkForBeat(ir)) return;

  long now   = millis();
  long delta = now - lastBeatMs;

  // BUG FIX 3: first beat only seeds the timestamp; skip BPM calculation.
  // Without this, delta = millis() - 0 = huge value → rejected or garbage BPM.
  if (lastBeatMs == 0) {
    lastBeatMs = now;
    return;
  }
  lastBeatMs = now;

  // Refractory window — reject physiologically impossible intervals
  if (delta < MIN_BEAT_MS || delta > MAX_BEAT_MS) return;

  float bpm = 60000.0f / (float)delta;

  // BUG FIX 4: outlier rejection — don't let motion artifacts corrupt the avg
  if (beatAvg > 20) {
    float deviation = fabsf(bpm - (float)beatAvg) / (float)beatAvg;
    if (deviation > OUTLIER_RATIO) return;
  }

  // Circular buffer
  rateBuf[rateBufIdx] = bpm;
  rateBufIdx = (rateBufIdx + 1) % RATE_BUF_SIZE;
  if (rateBufIdx == 0) rateBufFull = true;

  // Running average
  byte  count = rateBufFull ? RATE_BUF_SIZE : rateBufIdx;
  float sum   = 0;
  for (byte i = 0; i < count; i++) sum += rateBuf[i];
  beatAvg = (count > 0) ? (int)(sum / count) : 0;
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  Wire.begin();

  // ── MPU6050 ──────────────────────────────────────────────────────────────
  mpu.initialize();
  if (!mpu.testConnection()) {
    while (true) { Serial.println("ERR:MPU6050_NOT_FOUND"); delay(2000); }
  }
  mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);   // ±2 g → 16384 LSB/g
  mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_250);   // ±250 °/s
  mpu.setDLPFMode(MPU6050_DLPF_BW_20);              // 20 Hz low-pass

  // ── MAX30102 ─────────────────────────────────────────────────────────────
  if (!ppg.begin(Wire, I2C_SPEED_FAST)) {
    while (true) { Serial.println("ERR:MAX30102_NOT_FOUND"); delay(2000); }
  }
  /*
   * setup(brightness, sampleAvg, ledMode, sampleRate, pulseWidth, adcRange)
   *   brightness  60   → ~12 mA, good for fingertip
   *   sampleAvg    4   → 4 raw ADC readings averaged per FIFO entry
   *   ledMode      2   → Red + IR (checkForBeat uses IR channel)
   *   sampleRate 100   → 100 sps effective after averaging
   *                       = 5 samples per 50 ms output tick — FIFO safe
   *   pulseWidth 411   → 18-bit resolution (best SNR, per datasheet table 7)
   *   adcRange  4096   → 15.63 pA LSB (mid-range, matches typical finger signal)
   *
   * BUG FIX 5: removed setPulseAmplitudeRed/IR calls that overrode setup().
   * BUG FIX 2: reduced from 400 sps → 100 sps (sufficient for HR; prevents
   *             FIFO overflow between loop iterations on the Nano).
   */
  ppg.setup(60, 4, 2, 100, 411, 4096);

  // CSV header — helps downstream parsers (Python/MATLAB/Excel)
  Serial.println("timestamp_ms,ax,ay,az,ir,heart_rate_bpm");
}

void loop() {
  /*
   * BUG FIX 1 — drain the PPG FIFO on EVERY loop iteration.
   *
   * ppg.check()      pulls all queued hardware samples into the library buffer.
   * ppg.available()  returns how many unread samples are waiting.
   * ppg.getFIFOIR()  reads the IR value of the current sample.
   * ppg.nextSample() advances the read pointer (must call after each read).
   *
   * This decouples sensor sampling (100 Hz) from CSV output (20 Hz).
   * checkForBeat() now sees every sample in the correct sequence → stable BPM.
   */
  ppg.check();
  while (ppg.available()) {
    processPPGSample(ppg.getFIFOIR());
    ppg.nextSample();
  }

  // ── Output at 20 Hz ──────────────────────────────────────────────────────
  unsigned long now = millis();
  if (now - prevOutputMs < OUTPUT_INTERVAL_MS) return;
  prevOutputMs = now;

  int16_t rawAx, rawAy, rawAz, rawGx, rawGy, rawGz;
  mpu.getMotion6(&rawAx, &rawAy, &rawAz, &rawGx, &rawGy, &rawGz);

  float ax = rawAx / ACCEL_SCALE;
  float ay = rawAy / ACCEL_SCALE;
  float az = rawAz / ACCEL_SCALE;

  // lastIrValue was updated by the FIFO drain above — always fresh
  Serial.print(now);           Serial.print(',');
  Serial.print(ax, 4);         Serial.print(',');
  Serial.print(ay, 4);         Serial.print(',');
  Serial.print(az, 4);         Serial.print(',');
  Serial.print(lastIrValue);   Serial.print(',');
  Serial.println(beatAvg);
}