#!/usr/bin/env python3
"""
KHR-3HV 消費電力測定システム - PC側受信・表示スクリプト

ESP32から送られてくる CSV (timestamp_ms,voltage_V,current_A,clip_flag) を
シリアル経由でリアルタイム受信し、消費電力 P = V x I [W] を計算して
ターミナルに電圧/電流/電力を縦並びで表示する。

使い方:
    python3 power_monitor.py                      # ポート自動検出/選択
    python3 power_monitor.py --port /dev/cu.usbserial-XXXX
    python3 power_monitor.py --port COM3 --baud 115200
"""

import argparse
import signal
import sys
import time

import serial
import serial.tools.list_ports

DEFAULT_BAUD = 115200

# M5Stack UNIT INA226-10A (シャント抵抗5mΩ, 測定範囲±10A) の理論上の電流測定上限 [A]
SHUNT_FULL_SCALE_CURRENT_A = 10.0
CLIP_WARNING_RATIO = 0.95


def find_serial_port():
    """接続中のシリアルポートを検出する。複数あれば選択させる。"""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None
    if len(ports) == 1:
        return ports[0].device

    print("複数のシリアルポートが見つかりました:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device}  ({p.description})")
    while True:
        choice = input("使用するポート番号を選択してください: ").strip()
        try:
            idx = int(choice)
            if 0 <= idx < len(ports):
                return ports[idx].device
        except ValueError:
            pass
        print("無効な入力です。もう一度お試しください。")


def parse_args():
    parser = argparse.ArgumentParser(description="KHR-3HV 消費電力リアルタイム表示")
    parser.add_argument("--port", "-p", default=None,
                         help="シリアルポート (例: /dev/cu.usbserial-XXXX)")
    parser.add_argument("--baud", "-b", type=int, default=DEFAULT_BAUD,
                         help=f"ボーレート (デフォルト: {DEFAULT_BAUD})")
    return parser.parse_args()


class Display:
    """ターミナル上に電圧/電流/電力を縦並びで上書き表示する"""

    def __init__(self, port, baud):
        self.port = port
        self.baud = baud
        self.sample_count = 0
        self.start_time = time.monotonic()
        self._first_draw = True

    def _fmt_elapsed(self):
        elapsed = int(time.monotonic() - self.start_time)
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def update(self, voltage, current, clip_flag):
        self.sample_count += 1
        power = voltage * current

        warn = "  !! 電流が測定上限付近でクリップしています !!" if clip_flag else ""

        lines = [
            "KHR-3HV 消費電力モニター",
            f"Port: {self.port}   Baud: {self.baud}",
            "-" * 46,
            f"  電圧 [V] : {voltage:8.3f}",
            f"  電流 [A] : {current:8.3f}{warn}",
            f"  電力 [W] : {power:8.3f}",
            "-" * 46,
            f"Samples: {self.sample_count}   Elapsed: {self._fmt_elapsed()}",
            "Ctrl+C で終了",
        ]

        if self._first_draw:
            sys.stdout.write("\x1b[2J")  # 画面クリア
            self._first_draw = False

        sys.stdout.write("\x1b[H")  # カーソルをホームへ
        for line in lines:
            sys.stdout.write(line + "\x1b[K\n")
        sys.stdout.flush()

    def notice(self, message):
        # 通常表示を崩さないよう、画面最下部の下に追記する形で通知を出す
        sys.stdout.write(f"\n[ESP32] {message}\n")
        sys.stdout.flush()
        self._first_draw = True  # 次の更新でクリアし直す


def main():
    args = parse_args()

    port = args.port or find_serial_port()
    if not port:
        print("シリアルポートが見つかりませんでした。ESP32が接続されているか確認してください。",
              file=sys.stderr)
        sys.exit(1)

    print(f"接続中: {port} @ {args.baud}bps ...")

    try:
        ser = serial.Serial(port, args.baud, timeout=1)
    except serial.SerialException as e:
        print(f"シリアルポートを開けませんでした: {e}", file=sys.stderr)
        sys.exit(1)

    # ESP32がリセットされてブートするまで少し待つ
    time.sleep(2)
    ser.reset_input_buffer()

    display = Display(port, args.baud)

    def handle_sigint(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        while True:
            raw = ser.readline()
            if not raw:
                continue

            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            if line.startswith("#"):
                display.notice(line)
                continue

            fields = line.split(",")
            if len(fields) < 3:
                continue  # 不完全な行はスキップ

            try:
                float(fields[0])  # timestamp_ms (未使用だが値の妥当性チェックに使う)
                voltage = float(fields[1])
                current = float(fields[2])
            except ValueError:
                continue  # ヘッダ行など数値でない行はスキップ

            if len(fields) >= 4:
                clip_flag = fields[3].strip() == "1"
            else:
                clip_flag = abs(current) >= SHUNT_FULL_SCALE_CURRENT_A * CLIP_WARNING_RATIO

            display.update(voltage, current, clip_flag)

    except KeyboardInterrupt:
        print("\n\n終了処理中...")
    finally:
        ser.close()
        print("シリアルポートを閉じました。終了しました。")


if __name__ == "__main__":
    main()
