# 20 cm到達RL動作のWindows実機再生・電力計測

Task-only（λ=0）、R1/R2/R3（λ=0.3）の保存済み関節指令を、RCB-4搭載KHR-3HVへ50 Hzで送り、ESP32＋INA226の電圧・電流を同時記録します。対象は新規実験 `distance_goal_20261011` です。以前の20秒固定歩行とは異なるパッケージです。

**最初に `WINDOWS_HANDOFF_JA.txt` を読んでください。** WindowsではPython 3.11を使用します。Webots/PyTorchは不要です。

## 何を実機へ移すか

実験の凍結歩行生成器に保存された10 Hzの方策出力を入力し、遅延を含む17関節・50 Hzの位置指令を復元しました。全保存時点の指令・歩行パラメータと照合済みです。実測関節角度の単純補間ではありません。

これは **RLで得た関節指令のオープンループ再生** です。実機の観測をPPOに入力してリアルタイムで判断させる実装ではありません。実機の自動距離検出・20 cmでの自動停止も含みません。各動作はシミュレーションで20 cmへ到達した時点までの長さをそのまま再生します。実機では到達距離が変わるため、独立した距離計測が必要です。

再生終了時は最後の関節目標値を保持します。直立姿勢への遷移・脱力・電源遮断は自動実行しません。実機で終端姿勢を支えられるかは未検証です。停止後の安定姿勢への移行とそのエネルギーは、この実験の到達エネルギーに含めていません。

## 固定seed10の再生時間

| 条件 | 再生時間 |
|---|---:|
| Task-only | 12.1秒 |
| R1 | 9.4秒 |
| R2 | 9.5秒 |
| R3 | 9.5秒 |

## 同梱物

- `data/motions/`：4条件×6学習seed（10〜15）、評価seed20000固定の24動作。JSONは実行用、CSVは目視確認・他ソフト読込用。単位は秒・ラジアンです。
- `plan_pilot.json`：学習seed10固定、4条件×3反復＝12試行。各反復内の4条件をランダム順に配置済み。
- `plan_all_seeds.json`：任意の拡張用、4条件×6seed×3反復＝72試行。12試行と重複するseed10を独立データとして二重計上しないでください。
- `motion_catalog.csv`：各動作の長さ・Sim距離・予測エネルギー・OOD率。
- `motion_ranges.csv`：動作に必要な関節範囲と最大指令速度。実機の許容範囲を決める資料であり、機械的安全性を保証する値ではありません。
- `experiment.py` / `capture.py` / `common.py` / `analyze.py`：実行・保存・集計・作図。
- `hardware.example.json`：未校正の設定例。ID/SIOは候補、符号・ゼロ点等は未設定。
- `firmware/`：既存ESP32計測器から派生した、新規測定値だけを送るファームウェア。
- `vendor/`：デスクトップにあったRCBライブラリと電力受信コードの参照コピー。出典は `VENDOR_SOURCES.md`。
- `provenance/`：Sim監査、設計、各動作の元config/episode、凍結歩行生成器、Sim結果レポート。元のMacの絶対パスは来歴情報で、Windows再生時には参照しません。元レポート内の外部リンク先はこのZIPに全て含まれるわけではありません。
- `preview_pending_real/`：実際のSim値のみの図。Realは未測定。架空の実測値はありません。
- `tests/` / `VALIDATION.json`：実機を使わないソフトウェア検証とその記録。

## 12試行で十分か

**再生・測定系の動作確認と、固定した4動作の予備比較には使えます。論文で一般的な優位性を示すために十分と断言はできません。** 各条件1つの学習seedを3回再生する計画なので、評価できるのは同じ動作の実機反復ばらつきです。独立した3回の学習ではありません。

この12試行では全実測点と平均±標本SDを示し、有意差の星印を自動で付けません。失敗・短距離も報告します。学習seed間の再現性まで主張する場合は、他の学習seedも実機へ移す必要があります。72試行案も同梱しましたが、それ自体が十分な検出力を保証するわけではありません。確認的実験の必要数は、検出したい最小のエネルギー差、許容する失敗率、ばらつきと精度、多重比較を踏まえて事前に決めます。3反復の差だけを根拠に強い効果量を仮定しないでください。

参考：Lakens (2022), [Sample Size Justification](https://doi.org/10.1525/collabra.33267)。

## Windows設定

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
.\.venv\Scripts\python.exe experiment.py config --out hardware.json
.\.venv\Scripts\python.exe -m serial.tools.list_ports
```

`hardware.json` に実機で確認した情報を入力します。全17関節の `zero_pos`, `sign`（±1）, `units_per_rad`, `min_rad`, `max_rad`, `max_speed_rad_s`、ID/SIOを確認してください。

```text
servo_position = zero_pos + sign × simulator_angle_rad × units_per_rad
```

電力予測器の符号やWebotsの符号を、そのまま実機指令の符号として使わないでください。`mapping_verified=true` は確認後のみ設定します。HeartToHeart等のミキシングが重ならないことを確認し `mixing_off_verified=true` にします。実機の `max_current_A`, `min_voltage_V`, `max_initial_error_rad` と、校正済み電圧・電流係数 `power_calibration`、取得遅延・不確かさ `timing` を入力します。範囲外はクランプせず実行を拒否します。

ロボット用と電力計用のCOMポートは別です。競合するGUIを閉じてください。既存計測器は `legacy_receive` に対応しますが、古い値を新時刻で再送する可能性があります。同梱ファームウェアはその改善版です。実機でコンパイル・校正・更新確認後に `fresh_device` としてください。取得遅延を未検証で0と仮定しないでください。

50 Hz・`servo_frame=1` の通信を、支持した状態の試運転で確認してください。期限超過やACK異常は失敗として停止し、遅れた指令のまとめ送りはしません。通信が間に合わない際に再生速度を変更すると、別の動作になります。

## 1試行ずつ実行

12試行の順序は `trial_sheet.csv` と `plan_pilot.json` で確認できます。

```powershell
$plan = Get-Content .\plan_pilot.json -Raw | ConvertFrom-Json
$plan.trials | Format-Table trial_id,condition,training_seed,repeat
$trial = $plan.trials[0].trial_id

# ドライラン：ポートを開かず、出力フォルダも作らない
.\.venv\Scripts\python.exe experiment.py run --trial $trial --out "raw\$trial"

# 校正と初期姿勢、支持・停止手段を確認してから実行
.\.venv\Scripts\python.exe experiment.py run --trial $trial --out "raw\$trial" --execute
```

初期姿勢は各motion JSONの `initial_pose_rad` です。プログラムは初期位置を読み、誤差が大きければ動かさず拒否します。自動で姿勢合わせはしません。指令ACKは位置追従の保証ではなく、全関節の実測角度を運動中に記録する機能もありません。

2秒待機後に動作を再生し、その動作固有の時間が経過すると最後の目標を保持します。その後1秒まで電力記録を継続します。電力、送信・応答時刻、初期位置、エラー、計測窓をCSV/JSONへ保存します。Ctrl+Cは指令送信停止であってモーター電源遮断ではありません。物理停止手段を用意してください。姿勢・転倒の自動監視は実装していません。

毎回同じ床、初期姿勢、電池・電源条件、休止条件をそろえて記録します。電池電圧、温度、支持具の接触等は `trial_sheet.csv` に記載してください。支持具がロボットを持ち上げて負荷を変えた試行は、その事実を記録し自由歩行と混ぜないでください。失敗・欠測を隠さず、そのまま残します。出力フォルダは上書き不可です。

## 距離と到達時刻の記録

電力センサーでは距離を測れません。上方動画と校正した床目盛り等を用い、**同じ体幹の基準点**の初期前方軸方向変位を測ります。足先や総経路長ではありません。PCのSCORE START/END表示と動画を同期し、表示遅延も不確かさに含めます。自動カメラ追跡やハードウェア同期トリガーは含んでいません。

記録する量は以下です。

1. 指令再生開始から終了までの実測前進距離。
2. 20 cmに初めて達した時刻（開始からの秒数）と時刻不確かさ。
3. 転倒・横逸脱・中断などの有無。単に転倒して前へ伸びた状態を歩行成功としないでください。

下記は入力書式の例です。数値を必ず実測値に置き換えます。

```powershell
.\.venv\Scripts\python.exe experiment.py annotate --raw "raw\$trial" --distance-m 0.203 --outcome success --method "Synchronized overhead video; calibrated floor scale" --goal-time-s 9.1 --goal-time-uncertainty-s 0.05 --note "video filename; voltage; surface; uncertainty; operator"
```

`--goal-time-s` は平均速度から推定せず、実際の最初の20 cm通過を読み取ります。再生時間内の時刻のみ受け付けます。停止後の惰性・保持中の移動を、同じ到達動作に自動で含めません。

```powershell
# 20 cmに届かなかった例。到達時刻は入れない
.\.venv\Scripts\python.exe experiment.py annotate --raw "raw\$trial" --distance-m 0.16 --outcome shortfall --method "Calibrated overhead video" --note "20 cm not reached"
```

結果区分は `success / shortfall / fall / deviation / operator_stop / other_failure`。成功には実測前進距離20 cm以上が必要です。距離計測の不確かさで判定が曖昧な場合は断定せず注記します。到達時刻を測れなかった成功試行は、時刻オプションを省略できます。その場合は再生窓の集計だけに入り、20 cm到達エネルギーは未測定のままになります。

20 cm未達の動作を繰り返して距離を稼いだり、速度・振幅を実機結果に合わせて調整したりすると別実験になります。まず今回の動作をそのまま検証し、調整が必要なら元の失敗を残して新しい実験として計画します。

## 解析と論文用の図

```powershell
.\.venv\Scripts\python.exe experiment.py analyze --raw raw --out figures_R2 --sim-evaluator R2
```

- `sim_real_replay.pdf/svg/png`：上段は再生窓の平均前進速度、下段はその窓の積算エネルギー。**実機の距離が揃わない場合、同じ20 cm課題の省エネ比較とは呼べません。** 短距離・転倒でも計測が完了していれば残します。
- `sim_real_goal.pdf/svg/png`：独立に時刻を確認した成功試行の20 cm到達時間・到達エネルギー。成功試行への条件付けであるため、図内の成功数と全試行表も必ず報告してください。
- 電力は校正済みV×I、エネルギーは指定窓の境界を補間した台形積分。アイドル電力を差し引きません。停止後の保持・復帰エネルギーは含みません。
- Simの予測エネルギーは全4条件を共通R2予測器で評価します。R1/R3の補助図は `--sim-evaluator R1` / `R3` と新しい出力先で作れます。
- 12試行の図：青Simは各条件1本、誤差棒なし。赤Realは各条件3反復の平均±標本SDと全点。欠測時は存在する点を描き `DATA INCOMPLETE` と実数/予定数を表示します。
- 複数seed計画：各seed内の3反復が揃ったものを平均し、そのseed平均の間で平均±SDを計算。反復を独立した学習seedとして数えません。
- `all_trials.csv`：全予定試行、未実施・失敗・未達も含む。`seed_means.csv`：seedごとの集計。`plot_summary.csv`：図の数値と集計単位。`paired_seed_effects.csv`：同seed Task-onlyとの差（記述統計）。`analysis.json`：集計条件。

予備実験の英語キャプション例：
“Hardware replay of RL-derived 20-cm-goal trajectories (one preselected training seed per condition). Blue points show simulation values; all simulated energies use the common R2 predictor. Red points and error bars show the mean and sample standard deviation of three physical replays. Goal-energy results include only independently timed successful arrivals; success counts and all attempted trials are reported separately. Physical repetitions do not represent independent training seeds.”

## 実機なしの検証

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests
```

`--mock` は人工データで実機ポートを開きません。通常解析は拒否し、`--allow-mock` の場合だけ透かし付きのソフトウェア確認図を出します。Mockと実測のフォルダは分けてください。テストは一時フォルダのみ使用します。

本パッケージはMac上でソフトウェア検証済みです。Windowsでの実機通信、座標校正、安定性、計測器精度、派生ファームウェアのコンパイルは未検証です。Simの予測エネルギー低下を実機の省エネ実証とは呼びません。Sim全体では一部seedのOOD率が基準を超えた点も、同梱の元レポートに記録しています。
