# ROS 2・Gazeboによるロボット書道制御

毛筆風画像からロボットアームの軌道をBehavioral Cloning（BC）方策で学習し、強化学習（PPO）で追加学習する書道プロジェクトです。「一」「大」「小」の文字で学習し、未学習文字「学」を推論時に与え、既知のストローク軌道と目標画像に条件付けた描画制御を評価します。  

※ 松尾研 Physical AI応用1講座2026 最終課題 として取り組んだ。

## 実施結果

- 3文字共通のBC学習、BC初期化PPO、未学習文字「学」のオフライン比較を実装しました。
- ROS 2 Humble、MoveIt、Gazebo上で、方策推論から実測関節状態を使った全画再生まで実施しました。「学」は276/276ステップを完了しています。
- ストロークCSV、目標画像、実測軌跡から再生成した画像、実行条件を保存します。
- 最終発表資料とデモ動画の作成、コード整理は計画上未完了です。最新の進捗は[実施計画書](robot_calligraphy_training_plan.md)と[日次レポート](report/)を参照してください。

この成果はシミュレーション内の制御・画像評価です。画像ピクセルとGazebo world座標の独立な位置合わせ校正は未実施で、筆圧は実測力ではなくFKによるZ位置からの推定値です。紙面はvisual-onlyでcollisionを持たず、物理ブラシ・接触力モデルもありません。したがって、Gazeboの画像指標を実機の描画精度や物理接触性能として解釈しないでください。

## 構成

```text
fonts/{ichi,dai,sho}.csv + PNG
        |                     fonts/gaku.csv + PNG（最終評価のみ）
        v
brush_env.py -> bc_train.py -> rl_train.py -> evaluate_heldout.py
        |                                      |
        +-> font_renderer.py                  +-> 画像・指標・軌跡

ros2_calligraphy.launch.py
  Gazebo + robot_state_publisher + MoveIt + controllers + ros_gz_bridge
        ^
        | /joint_states、MoveIt FK/IK、FollowJointTrajectory
        |
ros2_policy_runner.py -> /paper/texture_update -> Gazebo paper texture
        +-> 実測CSV・target/measured画像・run_metadata.json
```

方策の観測は13要素、行動は基準軌道からのXY補正2要素と筆圧1要素です。Python環境の制御周期は20 Hz、XY補正上限は既定12 pxです。画間の移動は筆上げ状態として扱い、ストロークをまたいだ描画を行いません。

## 主なファイル

| ファイル | 役割 |
| --- | --- |
| `stroke_editor.py` | ストロークCSVと画像の作成・編集 |
| `validate_stroke_data.py` | CSV・画像ペアと座標データの検証 |
| `font_renderer.py` | 文字画像のレンダリング |
| `brush_env.py` | Gymnasium環境、筆跡生成、報酬・評価 |
| `bc_train.py` | ルールベース教師の生成と共通BC方策の学習 |
| `rl_train.py` | BC初期化PPO。ideal backendまたはGazebo feedbackに対応 |
| `evaluate_heldout.py` | 「学」のBaseline/BC/BC+RLオフライン比較 |
| `create_final_summary.py` | 4文字のオフライン比較画像・指標の生成 |
| `ros2_calligraphy.launch.py` | Gazebo、MoveIt、controllerなどの起動 |
| `ros2_policy_runner.py` | ROS 2閉ループ推論と実測結果の保存 |
| `ros2_gazebo_replay.md` | Gazebo起動・再生・評価の詳細手順 |
| `test_feedback_training.py` | 環境、筆圧写像、feedback処理のテスト |

## データ分割

| 用途 | 文字 | ファイル |
| --- | --- | --- |
| 学習・検証 | 一・大・小 | `fonts/ichi.csv` / `fonts/ichi.png`、`fonts/dai.csv` / `fonts/dai.png`、`fonts/sho.csv` / `fonts/sho.png` |
| 最終評価 | 学 | `fonts/gaku.csv` / `fonts/gaku.png` |

CSV列は`stroke_id,progress,x_px,y_px,z_press`です。`z_press`は現行データでは固定値で、熟練者の筆圧計測値ではありません。BC教師行動はCSVの値をそのまま教師とせず、現在画の軌道近傍と画像断面から推定します。「学」のCSV・画像はBC教師、RL報酬、正規化統計、モデル選択や条件調整には使わず、固定済み方策の推論入力と最終採点にのみ使います。

## 環境

オフライン学習・評価にはPythonと`requirements.txt`記載のパッケージが必要です。実績のあるロボットシミュレーション構成はUbuntu 22.04、ROS 2 Humble、MoveIt、Ignition Gazeboで、CRANE+ V2を使用しています。ROS 2の依存パッケージ、controller、Gazebo pluginを含む環境は別途用意してください。Windowsのこのworkspace単体ではROS 2/Gazeboの実行環境になりません。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

ROS 2の実行は、通常のPython venvではなくROS環境をsourceしたシェルで行います。実績のあるコンテナ内workspaceは`/home/ubuntu/pai_ws`です。起動条件、display、paper配置、controller確認の詳細は[Gazebo再生手順](ros2_gazebo_replay.md)を参照してください。

## 検証

データを確認し、テストを実行します。

```bash
python validate_stroke_data.py
python -m unittest test_feedback_training
```

計画書・日報では複数回のテスト成功が記録されています。追加された依存関係や環境差により結果が異なる場合があります。

## オフライン学習・評価

以下は現在のCLIに合わせた再実行例です。各工程の出力先を分け、評価対象の「学」を学習・調整に混ぜないでください。

```bash
python bc_train.py \
  --artifact-dir fonts \
  --output-dir artifacts/reproduction_bc \
  --seed 42

python rl_train.py \
  --motion-backend ideal \
  --bc-checkpoint artifacts/reproduction_bc/bc_policy.pt \
  --output-dir artifacts/reproduction_ppo \
  --episodes 30 \
  --episodes-per-update 3 \
  --seed 7

python evaluate_heldout.py \
  --data-dir fonts \
  --bc-checkpoint artifacts/reproduction_bc/bc_policy.pt \
  --rl-checkpoint artifacts/reproduction_ppo/bc_rl_policy.pt \
  --output-dir artifacts/reproduction_gaku
```

評価出力には比較画像、軌跡CSV、`metrics.json`が含まれます。BCとBC+RLの重み・前処理・条件を固定してからheld-out評価を行ってください。PPOの再学習結果はseedや実行条件で変動し、保存済み結果と同一になることを保証しません。

## ROS 2・Gazebo再生

1. Ubuntu環境でROS 2 Humble、Gazebo、MoveIt、controllerが利用可能であることを確認します。
2. [Gazebo再生手順](ros2_gazebo_replay.md)に従い、launch、controller、paper centerを設定します。現行SDFのpaper centerは`(0.19, 0.0) m`です。
3. 学習文字を使った短い動作確認の後、固定checkpointで`ros2_policy_runner.py`を実行します。
4. 出力ディレクトリの`run_metadata.json`で`character`、`error`、checkpointとmotion configを確認し、CSV・画像を評価します。

Gazebo PPOのcheckpointはpressure mapping、paper位置、meters-per-pixel、Z設定、endtip offsetなどと整合する必要があります。現在のpressure mappingは`paper_penetration_v1`です。古い設定のcheckpointは再利用せず、詳細手順に従ってください。ROS起動からrunner実行までの具体的なコマンドは頻繁に変更されるため、上記手順書を正とします。

## 保存済み結果

評価条件の異なる値を混ぜないよう、オフライン評価とGazebo実測を分けています。以下のDiceは画像類似度で、物理接触や実機性能を示す指標ではありません。

| 評価 | Baseline | BC | BC+RL | 出典・条件 |
| --- | ---: | ---: | ---: | --- |
| 「学」オフライン | 0.70761 | 0.69202 | 0.71110 | 計画書に記録された2026-10-30時点のoffline summary。3方式とも同一の目標画像で評価 |
| 「学」Gazebo実測軌跡からの再描画 | 0.71080 | 0.68168 | 0.73838 | `artifacts/evaluation/summary/metrics.json`。実測位置・FK Zからの推定圧力を使用 |

GazeboのBC+RLは40 mm/s、276ステップを完了し、metadataの`error`は`null`です。Gazebo評価の詳細な全文字指標は[評価metrics](artifacts/evaluation/summary/metrics.json)、個別再生条件は各`artifacts/evaluation/*/run_metadata.json`にあります。オフラインの既存比較画像は`artifacts/summary/`以下にもありますが、複数時点・条件の成果物が混在するため、画像とmetricsの更新時点・checkpointを照合して利用してください。

計画書に記録されたオフライン結果は単一seed・特定のレンダラ条件で、BC+RLの改善は小幅です。別途保存されたGazebo実測値は座標mapping未校正、物理ブラシ未実装の条件です。いずれも一般的な汎化性能やRLの優越性を確立する結果ではありません。

## 成果物・追加情報

- [研修課題 実施計画書](robot_calligraphy_training_plan.md): スコープ、設計、日程、達成判定、評価上の留保
- [ROS 2 / Gazebo リプレイ手順](ros2_gazebo_replay.md): 起動、外周テスト、再生、PPO feedback学習、トラブル対応
- [日次作業レポート](report/): 実験経過、条件、成果物、未確認事項
- [Gazebo paper texture plugin](gazebo_plugin/README.md): pluginの構築とtexture更新
- `artifacts/evaluation/`: 4文字×Baseline/BC/BC+RLのGazebo評価結果
- `artifacts/result/`: BC+RL checkpointと学習metrics
- `report/robot_calligraphy_report.html`: 図表を含む実施レポート

このworkspaceにある学習済みファイルは`artifacts/result/bc_rl_policy.pt`です。最新評価metadataが参照するBC checkpoint（`artifacts/day18_target_width_bc_20261101/bc_policy.pt`）およびGazebo PPO学習ディレクトリ一式は、このworkspaceのファイル一覧では確認できません。checkpointを再生・再現する際は、対応するBC/RL重みを復元し、metadataに記録された条件と一致させてください。

