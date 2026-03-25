#include <Wire.h>
#include <Adafruit_TCS34725.h>

Adafruit_TCS34725 tcs = Adafruit_TCS34725(TCS34725_INTEGRATIONTIME_50MS, TCS34725_GAIN_4X);

#define LED_PIN      13
#define SENSOR_LED   6

#define SAMPLE_COUNT     20      // number of readings to average
#define SAMPLE_DELAY_MS  250     // ms between each reading → 20 × 250 = 5 000 ms total
#define LOW_CONFIDENCE_DIST  20.0

struct GlucoseRef {
  const char* label;
  uint8_t r, g, b;
  bool isNoStrip;
};

const GlucoseRef LEVELS[] = {
  { "!! NO STRIP DETECTED !!",                119,  82,  56,  true  },
  { "Normal — No glucose detected",           104,  87,  56,  false },
  { "Low glucose    (~100 mg/dL)   (+)",      108,  86,  55,  false },
  { "Medium glucose (~250 mg/dL)   (++)",     114,  82,  50,  false },
  { "High glucose   (~500+ mg/dL)  (+++)",    117,  80,  50,  false }
};
const int NUM_LEVELS = sizeof(LEVELS) / sizeof(LEVELS[0]);

float colourDistance(int r1, int g1, int b1, int r2, int g2, int b2) {
  return sqrt(pow(r1 - r2, 2.0) + pow(g1 - g2, 2.0) + pow(b1 - b2, 2.0));
}

int normalise(uint16_t channel, uint16_t clear) {
  if (clear == 0) return 0;
  return constrain((int)((float)channel / clear * 255.0), 0, 255);
}

void setLEDs(bool state) {
  digitalWrite(LED_PIN,    state ? HIGH : LOW);
  digitalWrite(SENSOR_LED, state ? HIGH  : LOW);
}

void setup() {
  Serial.begin(9600);
  while (!Serial);

  pinMode(LED_PIN,    OUTPUT);
  pinMode(SENSOR_LED, OUTPUT);
  setLEDs(false);

  if (!tcs.begin()) {
    Serial.println(F("ERROR: TCS34725 not found! Check SDA/SCL wiring."));
    while (1);
  }

  Serial.println(F("========================================="));
  Serial.println(F("     Urine Glucose Strip Reader"));
  Serial.println(F("========================================="));
  Serial.println(F("Place the glucose pad of the strip in front"));
  Serial.println(F("of the sensor, then press ENTER to begin."));
  Serial.println(F("-----------------------------------------"));

  while (!Serial.available());
  while (Serial.available()) Serial.read();

  Serial.println(F("\n[1/3] Both LEDs ON — illuminating strip..."));
  setLEDs(true);
  delay(500);   // short settle before sampling begins

  // ── 5-second averaging loop ──────────────────────────────────────────────
  Serial.print(F("[2/3] Averaging "));
  Serial.print(SAMPLE_COUNT);
  Serial.println(F(" readings over 5 seconds..."));

  long sumR = 0, sumG = 0, sumB = 0, sumC = 0;

  for (int i = 0; i < SAMPLE_COUNT; i++) {
    uint16_t r, g, b, c;
    tcs.getRawData(&r, &g, &b, &c);
    sumR += r;  sumG += g;  sumB += b;  sumC += c;

    // Print a live progress dot every 5 samples
    if ((i + 1) % 5 == 0) {
      Serial.print(F("  Sample "));
      Serial.print(i + 1);
      Serial.print(F(" / "));
      Serial.println(SAMPLE_COUNT);
    }
    delay(SAMPLE_DELAY_MS);
  }

  setLEDs(false);

  // Compute averages
  uint16_t avgR = sumR / SAMPLE_COUNT;
  uint16_t avgG = sumG / SAMPLE_COUNT;
  uint16_t avgB = sumB / SAMPLE_COUNT;
  uint16_t avgC = sumC / SAMPLE_COUNT;

  Serial.println(F("[3/3] Computing average..."));

  int rn = normalise(avgR, avgC);
  int gn = normalise(avgG, avgC);
  int bn = normalise(avgB, avgC);

  Serial.println(F("\n--- Averaged Sensor Reading ---"));
  Serial.print(F("Avg Raw RGBC  : R=")); Serial.print(avgR);
  Serial.print(F("  G="));               Serial.print(avgG);
  Serial.print(F("  B="));               Serial.print(avgB);
  Serial.print(F("  C="));               Serial.println(avgC);
  Serial.print(F("Avg Normalised: R=")); Serial.print(rn);
  Serial.print(F("  G="));               Serial.print(gn);
  Serial.print(F("  B="));               Serial.println(bn);

  float minDist = 1e9;
  int   bestIdx = 0;

  Serial.println(F("\n--- Matching against reference colours ---"));
  for (int i = 0; i < NUM_LEVELS; i++) {
    float d = colourDistance(rn, gn, bn,
                             LEVELS[i].r, LEVELS[i].g, LEVELS[i].b);
    Serial.print(F("  "));
    Serial.print(LEVELS[i].label);
    Serial.print(F("  →  dist = "));
    Serial.println(d, 1);
    if (d < minDist) { minDist = d; bestIdx = i; }
  }

  Serial.println(F("\n========================================="));
  Serial.println(F("            RESULT"));
  Serial.println(F("========================================="));

  if (LEVELS[bestIdx].isNoStrip) {
    Serial.println(F("  !! NO STRIP DETECTED !!"));
    Serial.println(F("  Place the glucose pad directly over"));
    Serial.println(F("  the sensor and reset to try again."));
  } else {
    Serial.print(F("  Glucose Level : "));
    Serial.println(LEVELS[bestIdx].label);
    Serial.print(F("  Match distance: "));
    Serial.println(minDist, 1);

    if (minDist > LOW_CONFIDENCE_DIST) {
      Serial.println(F("\n  WARNING: Low confidence match."));
      Serial.println(F("  Re-centre the strip pad and retry."));
    } else {
      Serial.println(F("\n  Good colour match."));
    }
  }

  Serial.println(F("========================================="));
  Serial.println(F("Reset the Arduino to take a new reading."));
}

void loop() {}