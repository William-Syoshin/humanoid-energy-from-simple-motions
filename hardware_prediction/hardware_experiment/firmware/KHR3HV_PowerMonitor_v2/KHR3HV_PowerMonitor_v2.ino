// Proposed v2 firmware: not flashed or hardware-validated.
// Timestamp is the software observation of a fresh INA226 update, not ADC conversion start.
// Conversion/averaging latency must still be characterized.
#include <Wire.h>
#include <M5UnitUnified.h>
#include <M5UnitUnifiedMETER.h>

// ---- I2C ピン設定 ----
static const int I2C_SDA_PIN = 21;
static const int I2C_SCL_PIN = 22;
static const uint32_t I2C_CLOCK_HZ = 400000;

// ---- サンプリング間隔 ----
// 100ms(10Hz)。動作1回(クロール前進など)は数秒程度を想定しているため、
// 台形則で積分した場合の誤差は実用上小さい。ただし歩容の切り替わり直後など
// 電流が急変する区間ではこの間隔だと変化を取りこぼす可能性があるので、
// 積分結果の精度が問題になる場合はまずこの値を50ms程度まで下げて比較すること
// (I2C読み取り+シリアル送信のオーバーヘッド的に、10ms以下は現実的でない)。
static const unsigned long SAMPLE_INTERVAL_MS = 100;

// ---- シリアル通信 ----
static const unsigned long SERIAL_BAUD = 115200;

// 配線の極性により電流の符号が逆になる場合は true に変更する
static const bool INVERT_CURRENT_SIGN = false;

m5::unit::UnitUnified Units;
m5::unit::UnitINA226_10A ina226Unit;

unsigned long lastSampleMillis = 0;
float latestVoltageV = 0.0f;
float latestCurrentA = 0.0f;
bool hasFreshData = false;
unsigned long lastUpdateMillis = 0;
unsigned long lastEmittedUpdateMillis = 0;

// ---- 計測区間の状態 ----
bool isMeasuring = false;
String serialLineBuffer = "";

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(200);

  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  Wire.setClock(I2C_CLOCK_HZ);

  if (!Units.add(ina226Unit, Wire) || !Units.begin()) {
    Serial.println("# ERROR: INA226 (M5Unit) not detected. Check wiring and I2C address (0x41).");
    while (true) {
      delay(1000);
    }
  }

  Serial.println("# FIRMWARE KHR3HV_POWER_V2");
  Serial.println("# TIMESTAMP fresh_update_observed_ms");
  Serial.println("# READY");
  Serial.println("# Waiting for START command...");
}

void loop() {
  Units.update();

  if (ina226Unit.updated()) {
    // ライブラリは mV / mA 単位で返す
    latestVoltageV = ina226Unit.voltage() / 1000.0f;
    latestCurrentA = ina226Unit.current() / 1000.0f;
    if (INVERT_CURRENT_SIGN) {
      latestCurrentA = -latestCurrentA;
    }
    lastUpdateMillis = millis();
    hasFreshData = true;
  }

  handleSerialCommands();

  unsigned long now = millis();
  if (now - lastSampleMillis < SAMPLE_INTERVAL_MS) {
    return;
  }
  lastSampleMillis = now;

  if (!hasFreshData || lastUpdateMillis == lastEmittedUpdateMillis) {
    return; // まだ最初の測定が完了していない
  }

  if (!isMeasuring) {
    return; // STARTコマンドを受けるまではデータ行を出力しない
  }

  Serial.print(lastUpdateMillis);
  Serial.print(',');
  Serial.print(latestVoltageV, 4);
  Serial.print(',');
  Serial.println(latestCurrentA, 4);
  lastEmittedUpdateMillis = lastUpdateMillis;
  hasFreshData = false;
}

// シリアルバッファに来ているコマンドを行単位で読み取り、処理する。
// 改行が来るまでは1文字ずつ蓄積するだけなので、loop()内で毎回呼んでも
// 他の処理をブロックしない。
void handleSerialCommands() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (serialLineBuffer.length() > 0) {
        processCommand(serialLineBuffer);
        serialLineBuffer = "";
      }
    } else {
      serialLineBuffer += c;
      // 想定外に長い行が来た場合の保険(通常のSTART/STOPでは発生しない)
      if (serialLineBuffer.length() > 32) {
        serialLineBuffer = "";
      }
    }
  }
}

void processCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();

  if (cmd.startsWith("SYNC ")) {
    // Echo identifier and device time. Host brackets the transaction with send/receive times.
    Serial.print("# SYNC ");
    Serial.print(cmd.substring(5));
    Serial.print(' ');
    Serial.println(millis());
  } else if (cmd == "START") {
    isMeasuring = true;
    Serial.print("# START ");
    Serial.println(millis());
    Serial.println("timestamp_ms,voltage_V,current_A");
  } else if (cmd == "STOP") {
    isMeasuring = false;
    Serial.print("# STOP ");
    Serial.println(millis());
  } else {
    Serial.print("# UNKNOWN_COMMAND ");
    Serial.println(cmd);
  }
}
