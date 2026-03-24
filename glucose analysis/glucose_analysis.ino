include <Wire.h>
#include <Adafruit_TCS34725.h>

Adafruit_TCS34725 tcs = Adafruit_TCS34725(TCS34725_INTEGRATIONTIME_50MS, TCS34725_GAIN_4X);

#define LED_PIN      13   // Arduino Leonardo built-in LED
#define SENSOR_LED   6    // TCS34725 LED pin → connect to D6 (active LOW)

struct GlucoseRef {
  const char* label;
  uint8_t r, g, b;
};

const GlucoseRef LEVELS[] = {
  { "Normal — No glucose detected",           151, 208, 199 },
  { "~100 mg/dL  ( ~5 mmol/L )  (+)",         130, 198, 151 },
  { "~250 mg/dL  (~15 mmol/L)  (++)",          106, 173,  78 },
  { "~500 mg/dL  (~30 mmol/L)  (+++)",         147, 128,   7 },
  { "~1000 mg/dL (~60 mmol/L)  (++++)",        151, 104,  50 },
  { ">=2000 mg/dL (~110 mmol/L) (++++)",       125,  79,  45 }
};
const int NUM_LEVELS = sizeof(LEVELS) / sizeof(LEVELS[0]);

float colourDistance(int r1, int g1, int b1, int r2, int g2, int b2) {
  return sqrt(pow(r1 - r2, 2.0) + pow(g1 - g2, 2.0) + pow(b1 - b2, 2.0));
}

int normalise(uint16_t channel, uint16_t clear) {
  if (clear == 0) return 0;
  return constrain((int)((float)channel / clear * 255.0), 0, 255);
}

// ── Turn both LEDs ON or OFF together ─────────────────────────────────────
void setLEDs(bool state) {
  digitalWrite(LED_PIN, state ? HIGH : LOW);
  digitalWrite(SENSOR_LED, state ? LOW : HIGH);  // TCS34725 LED is active LOW
}

void setup() {
  Serial.begin(9600);
  while (!Serial);

  pinMode(LED_PIN,    OUTPUT);
  pinMode(SENSOR_LED, OUTPUT);
  setLEDs(false);  // both off initially

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

  // ── Step 1: Both LEDs ON ─────────────────────────────────────────────────
  Serial.println(F("\n[1/3] Both LEDs ON — illuminating strip..."));
  setLEDs(true);

  // ── Step 2: 2-second settle ──────────────────────────────────────────────
  Serial.println(F("[2/3] Waiting 2 seconds for stable reading..."));
  for (int i = 2; i > 0; i--) {
    Serial.print(i);
    Serial.println(F("..."));
    delay(1000);
  }

  // ── Step 3: Read ─────────────────────────────────────────────────────────
  Serial.println(F("[3/3] Capturing colour..."));

  uint16_t r, g, b, c;
  tcs.getRawData(&r, &g, &b, &c);

  setLEDs(false);  // both off after capture

  int rn = normalise(r, c);
  int gn = normalise(g, c);
  int bn = normalise(b, c);

  Serial.println(F("\n--- Sensor Reading ---"));
  Serial.print(F("Raw RGBC  : R="));
  Serial.print(r); Serial.print(F("  G="));
  Serial.print(g); Serial.print(F("  B="));
  Serial.print(b); Serial.print(F("  C="));
  Serial.println(c);

  Serial.print(F("Normalised: R="));
  Serial.print(rn); Serial.print(F("  G="));
  Serial.print(gn); Serial.print(F("  B="));
  Serial.println(bn);

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

    if (d < minDist) {
      minDist = d;
      bestIdx = i;
    }
  }

  Serial.println(F("\n========================================="));
  Serial.println(F("            RESULT"));
  Serial.println(F("========================================="));
  Serial.print(F("  Glucose Level : "));
  Serial.println(LEVELS[bestIdx].label);
  Serial.print(F("  Match distance: "));
  Serial.println(minDist, 1);

  if (minDist > 55.0) {
    Serial.println(F("\n  WARNING: Low confidence match."));
    Serial.println(F("  Centre the strip pad over the sensor"));
    Serial.println(F("  or allow more strip reaction time."));
  } else {
    Serial.println(F("\n  Good colour match."));
  }

  Serial.println(F("========================================="));
  Serial.println(F("Reset the Arduino to take a new reading."));
}

void loop() {}
