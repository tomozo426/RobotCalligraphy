# ROS 2 / Gazebo リプレイ手順

CRANE+ V2をGazebo上で起動し、BC+RL方策で未学習文字「学」を全画描画する手順。

## 事前確認

- Ubuntu 22.04、ROS 2 Humble、`/home/ubuntu/airobot_ws` overlayを使用する。
- Gazebo GUIは通常launchで起動する。`ign gazebo -g` で軌道すると、重複windowや黒画面になったことがある。
- この環境のdisplay割当は`:0`がコンテナ内VNC、`:8`がWindows側のdisplay。Gazebo GUIには`:8`を推奨する。`:3`はX serverへ接続できなかった。
- launch前に、既存Gazeboプロセスは終了すること。
- 作業ディレクトリは`/home/ubuntu/pai_ws`。

## 起動

端末AでGazebo、robot state publisher、MoveIt、controllerを起動する。

```bash
cd /home/ubuntu/pai_ws
source /opt/ros/humble/setup.bash
source /home/ubuntu/airobot_ws/install/setup.bash
export DISPLAY=:8
ros2 launch ./ros2_calligraphy.launch.py
```

通常起動はOGRE2/ソフトウェアレンダリング（llvmpipe）を使う。OpenGL描画もコードは入っているがOGRE2だとエラーになる（原因未調査）ためOGRE1を使ったが、かなり遅い。

```bash
PAI_GAZEBO_GPU_RENDERING=1 ros2 launch ./ros2_calligraphy.launch.py
```

GazeboのEntity Treeに`crane_plus`が表示され、scene内にarmが見えることを確認する。

## Headless起動

学習時にGUI描画を省く場合は、同じROS環境で次のように起動する。

```bash
PAI_GAZEBO_HEADLESS=1 ros2 launch ./ros2_calligraphy.launch.py
```

Ignition server-only mode（`-s`）を使うため、Gazebo GUIと画面表示は起動しない。physics、`gz_ros2_control`、MoveIt、`/joint_states`は通常どおり使い、camera sensor用のserver rendererは維持する。headlessではGUI上のrobotやpaper textureを目視できないため、controller状態とROS topicで確認する。通常の描画付き起動へ戻す場合は`PAI_GAZEBO_HEADLESS`を設定せずlaunchする。

## Paper texture plugins

通常launch時に`gazebo_plugin/`内のserver/GUI pluginを自動buildして読み込みます。`/paper/texture_update`へ256x256 RGB8画像を送ると、camera viewとGazeboメイン画面のpaper textureが更新されます。構成や個別buildの詳細は[gazebo_plugin/README.md](gazebo_plugin/README.md)を参照してください。

## Controller確認

端末Bで同じROS環境をsourceして状態を確認する。

```bash
source /opt/ros/humble/setup.bash
source /home/ubuntu/airobot_ws/install/setup.bash
ros2 control list_controllers
ros2 action info /crane_plus_arm_controller/follow_joint_trajectory
```

`joint_state_broadcaster`、`crane_plus_arm_controller`、`crane_plus_gripper_controller`が`active`で、arm actionの`Action servers`が`1`ならリプレイを開始できる。

## 関節位置モニター

Gazeboとcontrollerを起動したまま、別端末で関節の実測位置と目標位置をリアルタイム表示できる。

```bash
cd /home/ubuntu/pai_ws
source /opt/ros/humble/setup.bash
source /home/ubuntu/airobot_ws/install/setup.bash
export DISPLAY=:8
python3 joint_position_monitor.py
```

グラフの青緑のバーは`/joint_states`から受け取る実測位置、オレンジのダイヤはarm/gripper controllerの`reference`目標位置を示す。薄青の帯が可動範囲、赤い目盛りが上下限で、値の単位はrad。目標値が届く前も実測位置は表示される。

## 紙面外周テスト

アームの到達可能性確認のため紙の外周の連続移動をGazeboで確認する。lift高さで1周した後、紙面高さで1周し、開始位置でliftして終了。既定は紙面中心`(0.19, 0.0) m`、紙端から10 mm内側、5 mm間隔、15 mm/s。紙面高さの軌跡をGazebo textureにも表示する場合は`--draw`を付ける。これは画像表示用で、接触力や実際の筆圧を再現しない。

Gazebo起動後、controllerがactiveであることを確認し、別端末で実行する。

```bash
cd /home/ubuntu/pai_ws
source /opt/ros/humble/setup.bash
source /home/ubuntu/airobot_ws/install/setup.bash
python3 ros2_paper_perimeter_test.py --dry-run --draw
python3 ros2_paper_perimeter_test.py --draw
```

描画せずに動作だけ確認する場合は`--draw`を外す。1周の点数や開始角は`--spacing-mm`、`--start-corner`で変更できる。速度を変える必要がある場合のみ`--speed-mm-s`を指定する。

### 紙面位置の調整

紙面とcameraは`calligraphy_world.sdf`の`paper`モデルと`paper_camera`モデルの`<pose>`で配置する。現在は両方のX/Yが`(0.19, 0.0) m`。位置を変える場合は両モデルのX/Yを同じ値にし、外周テストでも同じ中心を指定する。

```bash
python3 ros2_paper_perimeter_test.py --paper-center-x 0.18 --paper-center-y 0.01 --draw
```

全画runnerを使う場合も`--paper-center-x`と`--paper-center-y`を同じ値にする。runnerの紙面中心X既定値は`0.10 m`なので、現在のSDF配置で実行するコマンドには`--paper-center-x 0.19 --paper-center-y 0.0`を付ける。SDFと各コマンドの中心座標が違うと、ロボットの軌道と紙面・textureの位置がずれる。

## 全画リプレイ

端末Bで方策推論を実行する。`--character`は`ichi`、`dai`、`sho`、`gaku`を選択でき、既定はheld-out文字の`gaku`。`gaku`にはBC+RL、学習文字にはBC方策を使う。毎回別の出力先を作る。

```bash
cd /home/ubuntu/pai_ws
source /opt/ros/humble/setup.bash
source /home/ubuntu/airobot_ws/install/setup.bash
output_dir="artifacts/day16_paper_penetration_replay_$(date +%Y%m%d_%H%M%S)"
rl_checkpoint="artifacts/day16_paper_penetration_ppo_40mms_20261030_063508/bc_rl_policy.pt" # paper_penetration_v1 checkpoint
python3 ros2_policy_runner.py --policy bc_rl --character gaku --paper-center-x 0.19 --paper-center-y 0.0 --pressure-z-span-mm 15 --z-transition-steps 3 --arm-z-offset-mm 0 --max-speed-mm-s 40 --reset-speed-mm-s 50 --bc-checkpoint artifacts/day10_endpoint_weight_bc/bc_policy.pt --rl-checkpoint "$rl_checkpoint" --output-dir "$output_dir"
```

学習文字「大」を描く場合はBC方策と`--character dai`を指定する。

```bash
output_dir="artifacts/day12_dai_bc_$(date +%Y%m%d_%H%M%S)"
python3 ros2_policy_runner.py --policy bc --character dai --bc-checkpoint artifacts/day10_endpoint_weight_bc/bc_policy.pt --paper-center-x 0.19 --paper-center-y 0.0 --output-dir "$output_dir"
```

runnerは開始時に筆先を最初の筆画点のlift位置へ移動し、その後ストロークを開始する。実測位置と実測Zから推定した筆圧で生成した筆跡画像は各制御stepで`/paper/texture_update`へRGB8 publishされる（既定5 Hz）。Gazeboのpaper textureは描画の進行に合わせて更新される。無効化する場合は`--texture-topic ""`を指定する。

`--arm-z-offset-mm`はarmの全Z目標へ加える補正で、負値は紙側へ下げる。FKの実測Zから同じ補正を差し戻してpressureを計算する。Gazebo checkpointでは学習時と同じ値を指定する。

pressure Z mappingは筆上げ高さと筆圧による紙面への押し込みを分離している。現行`paper_penetration_v1`では、接触中はpressure 0で`--contact-z`（紙面、既定0.0005 m）、pressure 1で`--contact-z - --pressure-z-span-mm`（既定15 mm押し込み、Z=`-0.0145 m`）へ線形に移動する。非接触中はpressure値に関係なく`--lift-z`（既定0.0205 m）を使い、接触状態の切替は`--z-transition-steps`（既定3 step）で補間する。Gazebo PPO checkpointはmapping version、span、transition steps、Z offset、endtip offset、command/max speedの一致を検査する。旧`contact_span_v1` checkpointは現行写像では使用できず、再学習が必要。

MoveItの`crane_plus_link_endtip`は可視brush meshの先端に合わせ、共有設定`ENDTIP_OFFSET_Z_M=0.176 m`（link4からのZ offset）を使う。このoffsetを変更した場合はrobot descriptionを再起動し、checkpointの`motion_config.endtip_offset_z_m`と一致する新checkpointを使う。現行は以前の0.178 m geometryで学習したcheckpointを再利用しない。

完了すると、選択文字の軌道CSV、目標画像、実測位置から再描画した画像、metadataが出力される。ファイル名は文字名のprefix付き（例: `dai_measured.png`）。metadataで`character`が選択文字と一致し、`error`が`null`であることを確認する。`steps_saved`は文字データにより異なり、「大」は180、「学」は276。

このコマンドは固定checkpointで方策推論を再実行する。保存済みCSVをそのまま軌道再生する機能ではない。筆幅は実測Zから推定した仮想筆圧から画像上で計算する。BaselineはCSV由来の補間軌道に目標画像の中心オフセットを適用し、目標筆幅6 pxを指令する。軌道CSVには中心補正済みreferenceと方策に渡す元のreferenceを別々に記録する。接触力を測る物理ブラシモデルではない。

## Gazebo実測フィードバックPPO学習

既定のPPO学習は従来どおり理想位置環境で動作する。Gazeboの`/joint_states`実測を使う場合は、Gazebo、MoveIt、controllerを起動し、別端末で次を実行する。学習は単一ロボット上で逐次実行され、episode reset時は現在位置でliftしてから開始位置へ移動する。

まず1step smokeで接続と計測を確認する。この設定ではBC/RL決定論評価を省き、1文字・1stepだけ実行する。Diceやlossは性能評価に使わない。

```bash
cd /home/ubuntu/pai_ws
source /opt/ros/humble/setup.bash
source /home/ubuntu/airobot_ws/install/setup.bash
output_dir="artifacts/day13_gazebo_feedback_pilot_$(date +%Y%m%d_%H%M%S)"
python3 rl_train.py --motion-backend gazebo \
   --bc-checkpoint artifacts/day10_endpoint_weight_bc/bc_policy.pt \
   --output-dir "$output_dir" --episodes 1 --episodes-per-update 1 \
   --max-steps 1 --skip-evaluation
```

`--max-steps`は各episodeを正常なtruncationとして終え、PPO valueをbootstrapする。0なら軌道全体を実行する。Gazebo学習速度の既定値は15 mm/sで、変更する場合は`--max-speed-mm-s`を指定する（例: `--max-speed-mm-s 40`）。pressureは`[0,1]`で正規化し、接触中はpaper surfaceから0〜`--pressure-z-span-mm`だけ下方向へ押し込む。指令後はaction完了に加え、新しい`/joint_states` sequenceと進んだROS stamp（stampが0ならsequence）を待ち、実測関節角をMoveIt FKする。実測Zを同じ線形写像で圧力へ戻し、その実測圧力を観測・筆幅・描画報酬に使う。入力CSVの`z_press`は要求せず、位置軌道はXYのみを使う。Baselineの目標幅は6 pxで、CSV値とは独立している。CSVとPPO metricsは指令要求、送信位置、実測位置・Z・圧力、joint-state freshness、simulation/wall rateを分けて保存する。`ik_residual_m`は要求XYZとIK解FKの差、`tracking_error_m`はIK解FKと実測FKの差。旧Gazebo PPO checkpointはpressure mapping versionが一致しないため再利用できず、新しいmappingで再学習する。

1step smokeが正常終了した後にのみ、3文字×20stepの動作pilotへ進む。決定論評価の追加移動を避けるため、このpilotでも`--skip-evaluation`を付ける。

```bash
output_dir="artifacts/day13_gazebo_feedback_pilot_$(date +%Y%m%d_%H%M%S)"
python3 rl_train.py --motion-backend gazebo \
   --bc-checkpoint artifacts/day10_endpoint_weight_bc/bc_policy.pt \
   --output-dir "$output_dir" --episodes 3 --episodes-per-update 3 \
   --max-steps 20 --skip-evaluation
```

smokeとpilotが正常終了した後、新しい`paper_penetration_v1` checkpointを学習する。以下はseed 7、30 episode、40 mm/sの実行例。再生時もcheckpointと同じmotion configを指定する。

```bash
output_dir="artifacts/day16_paper_penetration_ppo_40mms_$(date +%Y%m%d_%H%M%S)"
python3 rl_train.py --motion-backend gazebo \
   --bc-checkpoint artifacts/day10_endpoint_weight_bc/bc_policy.pt \
   --output-dir "$output_dir" --episodes 30 --episodes-per-update 3 --seed 7 \
   --max-speed-mm-s 40 --reset-speed-mm-s 50 \
   --paper-center-x 0.19 --paper-center-y 0.0 \
   --pressure-z-span-mm 15 --z-transition-steps 3 \
   --arm-z-offset-mm 0 --lift-z 0.0205
```

IK到達不能は描画せず固定ペナルティの失敗終端として記録する。joint state timeout、MoveIt service/action異常はepisodeを成功扱いせず学習を中断する。実測モードは指令完了後の離散測定であり、固定周期中の遅れや筆接触力を学ぶものではない。初回pilot後にGazebo動作、座標配置、freshness、所要時間、追従誤差を確認してからepisode数を増やす。

## 全4文字比較サマリ生成

PPO学習後、Baseline/BC/BC+RLのoffline rolloutをichi、dai、sho、gakuの4文字で実行し、比較画像を生成する。Gazeboを動かすコマンドではない。checkpointと出力先を指定する。

```bash
cd /home/ubuntu/pai_ws
bc_checkpoint="artifacts/day10_endpoint_weight_bc/bc_policy.pt"
rl_checkpoint="artifacts/day16_paper_penetration_ppo_40mms_20261030_063508/bc_rl_policy.pt"
summary_dir="artifacts/day16_paper_penetration_ppo_40mms_20261030_063508_summary"
python3 create_final_summary.py \
   --bc-checkpoint "$bc_checkpoint" --rl-checkpoint "$rl_checkpoint" \
   --output-dir "$summary_dir"
```

`$summary_dir/final_images_all_characters.png`がTarget/Baseline/BC/BC+RLの4行×4列比較画像、`trajectories_all_characters.png`が軌跡比較画像。`metrics.json`でcharacter一覧、held-out gakuをtraining/tuningに使っていないこと、各policyのDiceを確認する。オフライン比較のbaselineと軌跡図のreferenceは、CSV軌道に目標画像の断面中心オフセットを適用したもの（補正量はXY上限で制限）で、baselineの筆幅は固定6 px。BC・BC+RLの方策入力に使う基準軌道はCSV由来の補間軌道のまま。別checkpointで再生成する場合は`rl_checkpoint`と`summary_dir`を対応する学習結果へ変更する。

## 複数windowが存在する場合

Gazebo GUI/serverとcalligraphy launch配下のMoveIt、controller、bridgeなどをまとめて停止する場合は、workspace rootで次を実行する。最初はdry-runで対象だけを表示し、問題なければ通常停止を実行する。

```bash
cd /home/ubuntu/pai_ws
./stop_calligraphy_gazebo.sh --dry-run
./stop_calligraphy_gazebo.sh
```

scriptはこのworkspaceの`ros2_calligraphy.launch.py`、`calligraphy_world.sdf`、専用GUI configを識別し、そのprocess treeに限定する。他workspaceのGazeboは対象にしない。通常はSIGINT、残存時はSIGTERMを送信する。さらに強制終了が必要な場合は、dry-runの対象一覧を確認した後に`--force`を付ける。

```bash
./stop_calligraphy_gazebo.sh --force
```

停止後は「起動」または「Headless起動」の手順からやり直す。広範囲のprocessを対象にする`killall`は使わない。

## 黒画面の場合

- 今回は`ign gazebo -g`を別起動した際、GUI windowは作成されたもののworldが描画されず、画面captureも黒だった。GUI-only processが既存simulationのsceneを正しく表示できなかった可能性が高いが、ログに描画エラーはなく、低レベルの原因までは特定できていない。
- GUI-only processを追加起動しない。黒画面や重複windowが出た場合は、上記「複数windowの場合」の手順で重複processを整理し、`ros2 launch ./ros2_calligraphy.launch.py`を1組だけ起動し直す。
- 再起動後、Entity Treeに`crane_plus`があり、scene内にarmが表示されることを確認する。GUI windowがあるだけではrobotのspawnやscene表示が成功した証拠にならない。

## 制約

- `calligraphy_world.sdf`のtableはvisual-onlyで、接触物理を検証しない。
- Gazebo paper textureの画像軸向きはcamera image上のmarkerで確認済み。実機での絶対位置・接触状態は未検証で、実測Zからのpressureは推定値（実力センサによる測定ではない）。
- このworkspace/mapを実機へそのまま転用しない。

## 直近の実行例

2026-10-26の単一GUI実行では全276点が完了し、metadataは`error: null`だった。出力は`artifacts/day10_replay_visible_final/`に保存されている。詳細な試行記録と評価値は[report/day10.md](report/day10.md)を参照。