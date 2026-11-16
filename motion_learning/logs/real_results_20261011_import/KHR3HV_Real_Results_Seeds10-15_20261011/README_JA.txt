KHR-3HV 実機計測結果 引き継ぎ / 2026-10-11

最初に RESULTS.html をブラウザーで開いてください。ネット接続は不要です。
RESULTS.json は分析用の全24試行の一覧です。数値は生データから再計算しました。

収録範囲
seed 10〜15 × R1/R2/R3/Baseline × 各1回 = 24試行。
Baseline = Task-only（電力項の重みλ=0）。R1〜R3条件はλ=0.3。
これらの名称は学習・生成された動作の条件を表し、実測電力の予測精度そのものではありません。
MuJoCo由来の保存済み位置指令を実機へ50 Hzで再生し、既存の電圧・電流計測系で約10 Hzの電力サンプルを収集しました。
seed10→11→12→13→14→15、各seed内はR1→R3→R2→Baselineの順です。物理試行の順序はランダム化していません。
各seed内の繰り返しは1回です。6 seedsは同一動作の6回反復とは異なります。

フォルダー
RESULTS.html                    見やすい結果一覧（今回の確定した再生区間の実測値）
RESULTS.json                    24試行のエネルギー・平均電力・時間・相対パス
verify_results.py               Python標準ライブラリのみでファイル照合と24試行の再積分
SHA256_MANIFEST.json             全同梱ファイルのSHA-256（この一覧自体を除く）
experiment/commissioning_raw/    今回の24試行の生データと記録
experiment/calibration/         姿勢確認・開始姿勢へ戻した操作のログ（本測定とは別）
experiment/data/                使用した24動作のJSON/CSV、元manifest
experiment/provenance/          Windowsに渡されたシミュレーション由来の設定・資料
experiment/*.py                 実際に使用・更新した計測、実行、確認コード
experiment/hardware*.json        各試行の関節対応・角度範囲・監視設定
experiment/plan*.json            実行計画。元の12/72試行計画も資料として保持
experiment/original_code/       変更前コードの保存版
archive/                        計測前のプレビュー。今回の実測結果として使用しないこと

試行フォルダーの内容
power.csv: PC受信時刻、機器時刻、電圧、電流、クリップフラグ。
commands.csv: 指令送信の時刻・応答・遅延。送信角度そのものはmotion.jsonに収録。
initial_feedback.csv: 実行前の関節位置のみ。動作中の連続した実測関節角度ではありません。
motion.json: 再生した17関節の位置指令列と開始姿勢。
config.json: 当該試行の設定スナップショット。
meta.json/trial.json: 計測区間、状態、条件、seed、ファイルハッシュ。
measurement_summary.json: 計測後の集計。seed10の一部にはseed列がないため、全件の統一一覧はRESULTS.jsonを使ってください。
operator_confirmation.json等: オペレーターの申告がある場合のみ保存。
元ファイルの絶対Windowsパスは履歴として保持しています。Macでの分析はRESULTS.jsonのraw_directory（相対パス）を使用してください。

計算方法
各サンプルの P = voltage_V * current_A。
meta.jsonのscore_start〜score_end（PC monotonic clock）を境界で線形補間し、台形積分してenergy_Jを計算。
mean_power_W = energy_J / duration_s。
開始前・終了後の記録、開始姿勢への戻し動作のエネルギーは含みません。
待機電力は差し引いていません。機器時刻とPC時刻の単調性、計測区間のカバー、クリップ、サンプル欠落を確認。
既存の電圧・電流値を利用し、今回の追加較正・ファームウェア変更は行っていません。
受信時刻補正は0秒、既存設定のタイミング不確かさは0.101秒。

重要な未測定項目と解釈
・ユーザー方針により距離は未測定。ファイルのtarget_distance_m=0.2はシミュレーション側の目標であり、実機で20 cm達成した証拠ではありません。
・各動作の時間が異なります。総エネルギーの差だけで同距離の省エネルギー性を主張しないでください。
・各試行の転倒・接触・手による支持を自動判定していません。通信完了と物理的課題達成は別です。
・seed10のR1では「正しく前に進んだ」とユーザー確認があります。他は個別の成功確認を補っていません。
・mapping_verified=false / commissioning=trueを保持。6脚ピッチの方向と開始姿勢、R1前進は確認しましたが、全軸の厳密なシミュレータ座標対応は未確定です。
・電力と指令は記録されていますが、今回の歩行中の連続関節フィードバック・IMUは記録していません。
・確認用上限は段階的に変更しました。各試行config.json、original_code、実行計画を参照してください。元の動作指令の速度・角度をクリップや引き延ばしで変更していません。
・全24試行の記録がcompleteであり、送信数、電力ログ、積分値を確認済みです。データの物理的成功を捏造・補完していません。

Macでの検証
1. ZIPを解凍して、このREADMEがあるフォルダーをターミナルで開く。
2. python3 verify_results.py
追加パッケージ不要です。これはファイル照合と再計算のみで、ロボットへ接続しません。
experimentの実機操作コードはWindows COMポート設定です。Macでの分析には実行不要です。
同梱は今回Windowsで使用したデータ・コードと受領したシミュレーション資料です。Mac側の元の学習リポジトリ全体を含むものではありません。
旧README_JA.md / WINDOWS_HANDOFF_JA.txt / VALIDATION.jsonは受領時資料であり、今回変更後の最終結果・検証状態の説明ではありません。
今回の入口は本README、RESULTS.html、RESULTS.json、verify_results.pyです。
