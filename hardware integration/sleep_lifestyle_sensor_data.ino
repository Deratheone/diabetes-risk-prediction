/*
 * Combined Sensor Streamer — Arduino Nano
 * Hardware : Arduino Nano + MPU6050 (accel) + MAX30102 (PPG/IR)
 * Output   : timestamp,ax,ay,az,ir,heart_rate  @ 20 Hz over Serial
 *
 * Libraries (install via Library Manager):
 *   - "MPU6050" by Electronic Cats (or Jeff Rowberg)
 *   - "SparkFun MAX3010x Pulse and Proximity Sensor Library"
 *   - Wire (built-in)
 */

#include <Wire.h>
#include <MPU6050.h>
#include "MAX30105.h"
#include "heartRate.h"

#define SERIAL_BAUD       115200
#define SAMPLE_INTERVAL   50
#define ACCEL_SCALE       16384.0
#define RATE_BUFFER_SIZE  4
#define IR_FINGER_THRESH  50000

MPU6050  mpu;
MAX30105 ppgSensor;

byte  rateBuffer[RATE_BUFFER_SIZE];
byte  rateBufferIdx  = 0;
float beatsPerMinute = 0;
int   beatAvg        = 0;
long  lastBeatTime   = 0;
unsigned long previousMillis = 0;

void setup() {
  Serial.begin(SERIAL_BAUD);
  Wire.begin();

  mpu.initialize();
  if (!mpu.testConnection()) {
    while (true) { Serial.println("ERR:MPU6050_NOT_FOUND"); delay(2000); }
  }
  mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);
  mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_250);
  mpu.setDLPFMode(MPU6050_DLPF_BW_20);

  if (!ppgSensor.begin(Wire, I2C_SPEED_FAST)) {
    while (true) { Serial.println("ERR:MAX30102_NOT_FOUND"); delay(2000); }
  }
  ppgSensor.setup(60, 4, 2, 400, 411, 4096);
  ppgSensor.setPulseAmplitudeRed(0x0A);
  ppgSensor.setPulseAmplitudeIR(0x1F);
  delay(500);
}

int updateHeartRate(long irValue) {
  if (irValue < IR_FINGER_THRESH) {
    rateBufferIdx = 0; beatsPerMinute = 0; beatAvg = 0; lastBeatTime = 0;
    return 0;
  }
  if (checkForBeat(irValue)) {
    long now   = millis();
    long delta = now - lastBeatTime;
    lastBeatTime = now;
    if (delta > 300 && delta < 2000) {
      beatsPerMinute = 60.0 / (delta / 1000.0);
      rateBuffer[rateBufferIdx++] = (byte)beatsPerMinute;
      rateBufferIdx %= RATE_BUFFER_SIZE;
      int sum = 0;
      for (byte i = 0; i < RATE_BUFFER_SIZE; i++) sum += rateBuffer[i];
      beatAvg = sum / RATE_BUFFER_SIZE;
    }
  }
  return beatAvg;
}

void loop() {
  unsigned long now = millis();
  if (now - previousMillis < SAMPLE_INTERVAL) return;
  previousMillis = now;

  int16_t rawAx, rawAy, rawAz, rawGx, rawGy, rawGz;
  mpu.getMotion6(&rawAx, &rawAy, &rawAz, &rawGx, &rawGy, &rawGz);

  float ax = rawAx / ACCEL_SCALE;
  float ay = rawAy / ACCEL_SCALE;
  float az = rawAz / ACCEL_SCALE;

  long irValue = ppgSensor.getIR();
  int  hr      = updateHeartRate(irValue);

  Serial.print(now);      Serial.print(',');
  Serial.print(ax, 4);    Serial.print(',');
  Serial.print(ay, 4);    Serial.print(',');
  Serial.print(az, 4);    Serial.print(',');
  Serial.print(irValue);  Serial.print(',');
  Serial.println(hr);
}