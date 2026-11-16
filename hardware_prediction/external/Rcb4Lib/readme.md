
# Rcb4Lib for Python
---
## Overview
近藤科学製RCB4をPythonで動かすためのライブラリです。
This library is for connecting a Kondo Kagaku RCB4 from an Arudino serial(HardwareSerial) port.

## Description
近藤科学製RCB4(ロボット用コントロールボード)をPythonで動かすためのライブラリです。  
This library is for connecting a RCB4 (Kondo kagaku robot control board) using Python.

Pythonのライブラリからから
- RCB4内に保存されているのモーションデータを再生
  Motion data preserved by RCB4 is played.

- Rcb4に接続されているサーボモータに角度指令を送ることができる
  Posiotn order can be sent to servo motor which leads to RCB4.

...etc


## Requirement
Python 3  (check to Python3.5.3)

- serial(pySerial)
- Enum
- struct


## Usage
配布フォルダのマニュアルをご覧ください。
For details, refer to the manual (PDF) in the folder provided.

関数一覧は、配布フォルダのFunctionListをご覧ください。
For details,Rcb4Lib function list in the folder(FunctionList) provided.

## Licence
Copyright 2019 Kondo Kagaku co.,ltd.
[MIT License](http://opensource.org/licenses/mit-license.php)
see to MIT_Licence.txt


## Author
近藤科学株式会社
Kondo Kagaku co.,ltd.
T.Nobuhara
近藤科学ホームページ:(<http://kondo-robot.com/>)
