# Gazebo texture plugins

Gazeboのpaper visualへ画像を反映するserver pluginとGUI plugin、およびそのbuild resourceをまとめたフォルダです。

## ファイル

- `texture_update_probe.cc`: server側のGazebo System plugin。`/paper/texture_update`の画像をserverのOGRE2 textureへuploadし、camera renderingへ反映します。
- `texture_update_gui.cc`: GUI側plugin。同じ画像をGUI process内のOGRE2 textureへuploadし、Gazeboメイン画面へ反映します。
- `TextureUpdateGui.qml` / `TextureUpdateGui.qrc`: GUI pluginを読み込むための非表示QML resourceです。
- `build_texture_update_probe.sh`: 2つのshared libraryとQML resourceをbuildします。

## Buildと起動

workspace rootからbuildします。

```bash
bash ./gazebo_plugin/build_texture_update_probe.sh
```

成果物はworkspaceの`build/`へ出力されます。通常は`ros2_calligraphy.launch.py`がlaunch時にこのscriptを実行するため、個別buildは不要です。

```bash
source /opt/ros/humble/setup.bash
source /home/ubuntu/airobot_ws/install/setup.bash
export DISPLAY=:8
ros2 launch ./ros2_calligraphy.launch.py
```

## Image contract

両pluginはROS-Gazebo bridgeから`ignition.msgs.Image`として届く`/paper/texture_update`を購読します。画像は256x256のRGB8を想定し、server側uploadはcamera view、GUI側uploadはmain 3D viewを更新します。launchがtopic bridgeとplugin探索pathを設定します。