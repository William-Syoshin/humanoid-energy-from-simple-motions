// Derived for transfer acquisition; not compiled or hardware-validated here.
#include <cstring>
/*
 * KHR-3HV 消費電力測定システム - ESP32側スケッチ (M5Stack UNIT INA226-10A版)
 *
 * ハードウェア:
 *   - ESP32 (Freenove ESP32 WROOM)
 *   - M5Stack UNIT INA226-10A (絶縁型, シャント抵抗5mΩ固定, 測定範囲±10A, 電圧範囲0-30V)
 *     I2C: SDA -> GPIO21, SCL -> GPIO22, 電源 5V
 *     https://docs.m5stack.com/en/unit/Unit_INA226-10A
 *
 * 使用ライブラリ (M5Stack公式, Arduino Library Managerからインストール):
 *   "M5Unit-METER" (M5Stack)
 *   依存ライブラリ (Library Managerが自動解決): M5UnitUnified, M5Utility, M5HAL
 *   ※ 表示器(M5.Display)やM5.begin()を必要とする"M5Unified"本体は不要。
 *      本スケッチはWireのみでM5UnitUnifiedフレームワークを利用する。
 *
 * 重要な注意点:
 *   - このユニットのデフォルトI2Cアドレスは 0x41 (0x40ではない)。
 *     ライブラリの UnitINA226_10A クラスに組み込み済み。
 *   - シャント抵抗5mΩ・最大測定電流10AもUnitINA226_10Aクラスに内蔵済みのため、
 *     手動でのキャリブレーション計算は不要。
 *   - 双方向(bidirectional)測定対応のため、配線の極性によっては電流が負の値になる。
 *     壊れることはないので、負の値が出る場合は下記 INVERT_CURRENT_SIGN を true にするか、
 *     電力入出力端子の+/-を入れ替えること。
 *
 * シリアル出力形式 (CSV, 115200bps):
 *   1行目: ヘッダ "timestamp_ms,voltage_V,current_A"
 *   2行目以降: "12345,10.523,0.412"
 *   '#' で始まる行はエラー/情報メッセージ (Python側でコメント行として無視すること)
 */

#include <Wire.h>
#include <M5UnitUnified.h>
#include <M5UnitUnifiedMETER.h>

// ---- I2C ピン設定 ----
static const int I2C_SDA_PIN = 21;
static const int I2C_SCL_PIN = 22;
static const uint32_t I2C_CLOCK_HZ = 400000;

// ---- サンプリング間隔 ----
static const unsigned long SAMPLE_INTERVAL_MS = 100; // 100ms = 10Hz

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
unsigned long freshTimestampMs = 0;
char commandBuffer[16];
size_t commandLength = 0;
void announceContract() {
  Serial.println("# FIRMWARE KHR3HV_POWER_V2");
  Serial.println("# TIMESTAMP fresh_update_observed_ms");
  Serial.println("timestamp_ms,voltage_V,current_A,clip_flag");
}

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

  announceContract();
}

void loop() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      commandBuffer[commandLength] = 0;
      if (strcmp(commandBuffer, "START") == 0) announceContract();
      commandLength = 0;
    } else if (c != '\r' && commandLength < sizeof(commandBuffer)-1) {
      commandBuffer[commandLength++] = c;
    }
  }
  Units.update();

  if (ina226Unit.updated()) {
    // ライブラリは mV / mA 単位で返す
    latestVoltageV = ina226Unit.voltage() / 1000.0f;
    latestCurrentA = ina226Unit.current() / 1000.0f;
    if (INVERT_CURRENT_SIGN) {
      latestCurrentA = -latestCurrentA;
    }
    hasFreshData = true;
    freshTimestampMs = millis();
  }

  unsigned long now = millis();
  if (now - lastSampleMillis < SAMPLE_INTERVAL_MS) {
    return;
  }
  lastSampleMillis = now;

  if (!hasFreshData) {
    return; // まだ最初の測定が完了していない
  }

  Serial.print(freshTimestampMs);
  Serial.print(',');
  Serial.print(latestVoltageV, 4);
  Serial.print(',');
  Serial.print(latestCurrentA, 4);
  Serial.print(',');
  Serial.println((latestCurrentA >= 9.5f || latestCurrentA <= -9.5f) ? 1 : 0);
  hasFreshData = false; // Never re-stamp an old sample as a new measurement.
}
