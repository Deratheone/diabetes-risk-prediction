#include <Wire.h>
#include <MPU6050.h>
#include "MAX30105.h"
#include "heartRate.h"

#define SERIAL_BAUD         115200
#define OUTPUT_INTERVAL_MS  50
#define ACCEL_SCALE         16384.0f
#define RATE_BUF_SIZE       8
#define IR_FINGER_THRESH    50000L
#define MIN_BEAT_GAP_MS     333L
#define MAX_BEAT_MS         1500L
#define OUTLIER_TOLERANCE   0.35f

MPU6050  mpu;
MAX30105 ppg;

float rateBuf[RATE_BUF_SIZE];
byte  rateBufIdx  = 0;
bool  rateBufFull = false;
long  lastBeatMs  = 0;
int   beatAvg     = 0;
long  lastIrValue = 0;

unsigned long prevOutputMs = 0;

float bufferAverage() {
  byte count = rateBufFull ? RATE_BUF_SIZE : rateBufIdx;
  if (count == 0) return 0;
  float sum = 0;
  for (byte i = 0; i < count; i++) sum += rateBuf[i];
  return sum / count;
}

void resetHRState() {
  for (byte i = 0; i < RATE_BUF_SIZE; i++) rateBuf[i] = 70.0f;
  rateBufIdx  = 0;
  rateBufFull = false;
  lastBeatMs  = 0;
  beatAvg     = 70;
}

void processPPGSample(long ir) {
  lastIrValue = ir;

  if (ir < IR_FINGER_THRESH) {
    resetHRState();
    return;
  }

  if (!checkForBeat(ir)) return;

  long now   = millis();
  long delta = now - lastBeatMs;

  if (lastBeatMs == 0) {
    lastBeatMs = now;
    return;
  }

  lastBeatMs = now;

  if (delta < MIN_BEAT_GAP_MS || delta > MAX_BEAT_MS) return;

  float bpm = 60000.0f / (float)delta;
  if (bpm < 40.0f || bpm > 180.0f) return;

  float avg = bufferAverage();
  if (avg > 10.0f) {
    if (fabsf(bpm - avg) > avg * OUTLIER_TOLERANCE) return;
  }

  rateBuf[rateBufIdx] = bpm;
  rateBufIdx = (rateBufIdx + 1) % RATE_BUF_SIZE;
  if (rateBufIdx == 0) rateBufFull = true;

  beatAvg = (int)bufferAverage();
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(1000);

  Wire.end();
  delay(50);
  Wire.begin();
  Wire.setClock(100000UL);
  delay(250);

  // ── MPU6050 ──────────────────────────────────────────────────────────────
  Serial.println(F("Init MPU6050..."));
  mpu.initialize();
  delay(200);

  // Read WHO_AM_I directly — clone chips return 0x70/0x72/0x98
  // instead of 0x68, making testConnection() fail even though
  // the sensor works perfectly. We bypass testConnection() entirely
  // and just verify the register is readable.
  Wire.beginTransmission(0x68);
  Wire.write(0x75);
  Wire.endTransmission(false);
  Wire.requestFrom(0x68, 1);
  byte whoAmI = Wire.read();

  Serial.print(F("MPU6050 WHO_AM_I: 0x"));
  Serial.print(whoAmI, HEX);

  // Accept 0x68 (genuine) and common clones 0x70, 0x72, 0x98
  if (whoAmI == 0x68 || whoAmI == 0x70 ||
      whoAmI == 0x72 || whoAmI == 0x98) {
    Serial.println(F(" — OK (chip detected)"));
  } else {
    // Completely unrecognised — halt
    while (true) {
      Serial.println(F("ERR: MPU6050 not found — check SDA/SCL/VCC/GND"));
      delay(2000);
    }
  }

  mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);
  mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_250);
  mpu.setDLPFMode(MPU6050_DLPF_BW_20);
  Serial.println(F("MPU6050 OK"));

  // ── MAX30102 ─────────────────────────────────────────────────────────────
  Serial.println(F("Init MAX30102..."));
  if (!ppg.begin(Wire, I2C_SPEED_STANDARD)) {
    while (true) {
      Serial.println(F("ERR: MAX30102 not found — check SDA/SCL and 3.3V"));
      delay(2000);
    }
  }

  byte partID = ppg.readPartID();
  Serial.print(F("MAX30102 chip ID: 0x"));
  Serial.print(partID, HEX);
  if (partID == 0x15) {
    Serial.println(F("  (OK)"));
  } else {
    Serial.println(F("  (WARNING: unexpected ID)"));
  }

  ppg.setup(60, 4, 2, 100, 411, 4096);
  resetHRState();

  Serial.println(F("MAX30102 OK - LEDs are now active"));
  Serial.println(F("Place fingertip gently on sensor. Do not press hard."));
  Serial.println();
  Serial.println(F("timestamp_ms,ax,ay,az,ir,heart_rate_bpm"));
}

void loop() {
  processPPGSample(ppg.getIR());

  unsigned long now = millis();
  if (now - prevOutputMs < OUTPUT_INTERVAL_MS) return;
  prevOutputMs = now;

  int16_t rawAx, rawAy, rawAz, rawGx, rawGy, rawGz;
  mpu.getMotion6(&rawAx, &rawAy, &rawAz, &rawGx, &rawGy, &rawGz);

  float ax = rawAx / ACCEL_SCALE;
  float ay = rawAy / ACCEL_SCALE;
  float az = rawAz / ACCEL_SCALE;

  Serial.print(now);           Serial.print(',');
  Serial.print(ax, 4);         Serial.print(',');
  Serial.print(ay, 4);         Serial.print(',');
  Serial.print(az, 4);         Serial.print(',');
  Serial.print(lastIrValue);   Serial.print(',');
  Serial.println(beatAvg);
}