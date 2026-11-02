#!/usr/bin/env bash
set -euo pipefail

plugin_source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(dirname -- "${plugin_source_dir}")"
build_dir="${project_root}/build"
mkdir -p "${build_dir}"

g++ -std=c++17 -shared -fPIC \
  -I/usr/include/ignition/gazebo6 \
  -I/usr/include/ignition/rendering6 \
  -I/usr/include/ignition/plugin1 \
  -I/usr/include/ignition/common4 \
  -I/usr/include/ignition/math6 \
  -I/usr/include/ignition/sdformat12 \
  -I/usr/include/ignition/msgs8 \
  -I/usr/include/ignition/transport11 \
  -I/usr/include/ignition/utils1 \
  -I/usr/include/ignition/cmake2 \
  -I/usr/include/OGRE-Next \
  -I/usr/include/OGRE-Next/Hlms/Pbs \
  -I/usr/include/OGRE-Next/Hlms/Common \
  -o "${build_dir}/libtexture_update_probe.so" \
  "${plugin_source_dir}/texture_update_probe.cc" \
  -lignition-gazebo6 \
  -lignition-gazebo6-rendering \
  -lignition-rendering6-ogre2 \
  -lignition-rendering6 \
  -lignition-plugin1 \
  -lignition-common4 \
  -lignition-math6 \
  -lignition-transport11 \
  -lsdformat12 \
  -lOgreNextMain

printf 'Built %s\n' "${build_dir}/libtexture_update_probe.so"

rcc -name TextureUpdateGui \
  -o "${build_dir}/TextureUpdateGui_qrc.cc" \
  "${plugin_source_dir}/TextureUpdateGui.qrc"

g++ -std=c++17 -shared -fPIC \
  -I/usr/include/ignition/gazebo6 \
  -I/usr/include/ignition/gui6 \
  -I/usr/include/ignition/rendering6 \
  -I/usr/include/ignition/plugin1 \
  -I/usr/include/ignition/common4 \
  -I/usr/include/ignition/math6 \
  -I/usr/include/ignition/sdformat12 \
  -I/usr/include/ignition/msgs8 \
  -I/usr/include/ignition/transport11 \
  -I/usr/include/ignition/utils1 \
  -I/usr/include/ignition/cmake2 \
  -I/usr/include/OGRE-Next \
  -I/usr/include/OGRE-Next/Hlms/Pbs \
  -I/usr/include/OGRE-Next/Hlms/Common \
  $(pkg-config --cflags Qt5Core Qt5Gui Qt5Widgets Qt5Qml Qt5Quick Qt5QuickControls2) \
  -o "${build_dir}/libTextureUpdateGui.so" \
  "${plugin_source_dir}/texture_update_gui.cc" \
  "${build_dir}/TextureUpdateGui_qrc.cc" \
  -lignition-gazebo6-gui \
  -lignition-gui6 \
  -lignition-rendering6-ogre2 \
  -lignition-rendering6 \
  -lignition-transport11 \
  -lignition-plugin1 \
  -lsdformat12 \
  -lOgreNextMain \
  $(pkg-config --libs Qt5Core Qt5Gui Qt5Widgets Qt5Qml Qt5Quick Qt5QuickControls2)

printf 'Built %s\n' "${build_dir}/libTextureUpdateGui.so"