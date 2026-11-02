import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import yaml
from ament_index_python.packages import get_package_share_directory
from crane_plus_description.robot_description_loader import RobotDescriptionLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calligraphy_config import ENDTIP_OFFSET_Z_M
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node, SetParameter


def load_file(package_name, file_path):
    path = os.path.join(get_package_share_directory(package_name), file_path)
    with open(path, encoding="utf-8") as file:
        return file.read()


def load_yaml(package_name, file_path):
    path = os.path.join(get_package_share_directory(package_name), file_path)
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def generate_launch_description():
    project_root = os.path.dirname(os.path.abspath(__file__))
    build_script = os.path.join(
        project_root, "gazebo_plugin", "build_texture_update_probe.sh"
    )
    subprocess.run(["bash", build_script], check=True, cwd=project_root)
    plugin_dir = os.path.join(project_root, "build")

    description_loader = RobotDescriptionLoader()
    description_loader.use_gazebo = "true"
    description_loader.use_ignition = "true"
    description_loader.gz_control_config_package = "crane_plus_control"
    description_loader.gz_control_config_file_path = "config/crane_plus_controllers.yaml"
    description_loader.robot_description_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "crane_plus_calligraphy.urdf.xacro",
    )
    description_loader.endtip_offset_xyz = f"0 0.03 {ENDTIP_OFFSET_Z_M:.3f}"
    robot_description = {"robot_description": description_loader.load()}
    robot_semantic = {
        "robot_description_semantic": load_file(
            "crane_plus_moveit_config", "config/crane_plus.srdf"
        )
    }
    kinematics = {"robot_description_kinematics": load_yaml(
        "crane_plus_moveit_config", "config/kinematics.yaml"
    )}
    kinematics["robot_description_kinematics"]["arm"]["position_only_ik"] = True
    ompl = {"move_group": {
        "planning_plugin": "ompl_interface/OMPLPlanner",
        "request_adapters": "default_planner_request_adapters/AddTimeOptimalParameterization "
        "default_planner_request_adapters/FixWorkspaceBounds "
        "default_planner_request_adapters/FixStartStateBounds "
        "default_planner_request_adapters/FixStartStateCollision "
        "default_planner_request_adapters/FixStartStatePathConstraints",
        "start_state_max_bounds_error": 0.1,
    }}
    ompl["move_group"].update(
        load_yaml("crane_plus_moveit_config", "config/ompl_planning.yaml")
    )
    controllers = load_yaml("crane_plus_moveit_config", "config/controllers.yaml")
    moveit_controllers = {
        "moveit_simple_controller_manager": controllers,
        "moveit_controller_manager":
            "moveit_simple_controller_manager/MoveItSimpleControllerManager",
    }

    gpu_rendering = os.environ.get("PAI_GAZEBO_GPU_RENDERING", "").lower() in {
        "1", "true", "yes", "on",
    }
    headless = os.environ.get("PAI_GAZEBO_HEADLESS", "").lower() in {
        "1", "true", "yes", "on",
    }
    render_engine = "ogre" if gpu_rendering else "ogre2"
    plugin_paths = [
        path for path in (plugin_dir, os.environ.get("LD_LIBRARY_PATH", "")) if path
    ]
    resource_paths = [
        project_root,
        os.path.dirname(get_package_share_directory("crane_plus_description")),
    ]
    env = {
        "IGN_GAZEBO_SYSTEM_PLUGIN_PATH": os.pathsep.join(plugin_paths),
        "IGN_GUI_PLUGIN_PATH": plugin_dir,
        "IGN_GAZEBO_RESOURCE_PATH": os.pathsep.join(resource_paths),
        "LIBGL_ALWAYS_SOFTWARE": "0" if gpu_rendering else "1",
    }
    world_file = os.path.join(project_root, "calligraphy_world.sdf")
    if gpu_rendering:
        gui_config = os.path.join(
            project_root, "gui_ogre.config"
        )
    else:
        gui_config = os.path.join(
            get_package_share_directory("crane_plus_gazebo"), "gui", "gui.config"
        )
    with open(gui_config, encoding="utf-8") as file:
        gui_config_xml = file.read()
    if gui_config_xml.lstrip().startswith("<?xml"):
        gui_config_xml = gui_config_xml[gui_config_xml.index("?>") + 2:]
    gui_config_root = ET.fromstring(f"<root>{gui_config_xml}</root>")
    gui_plugin = ET.SubElement(
        gui_config_root, "plugin",
        {"filename": "TextureUpdateGui", "name": "TextureUpdateGui"},
    )
    ignition_gui = ET.SubElement(gui_plugin, "ignition-gui")
    ET.SubElement(ignition_gui, "title").text = "Paper texture sync"
    ET.SubElement(ignition_gui, "property", {
        "type": "bool", "key": "showTitleBar",
    }).text = "false"
    ET.SubElement(ignition_gui, "property", {
        "type": "bool", "key": "resizable",
    }).text = "false"
    ET.SubElement(ignition_gui, "property", {
        "type": "double", "key": "x",
    }).text = "-1000"
    ET.SubElement(ignition_gui, "property", {
        "type": "double", "key": "y",
    }).text = "-1000"
    ET.SubElement(ignition_gui, "property", {
        "type": "double", "key": "width",
    }).text = "1"
    ET.SubElement(ignition_gui, "property", {
        "type": "double", "key": "height",
    }).text = "1"
    ET.SubElement(ignition_gui, "property", {
        "type": "string", "key": "state",
    }).text = "floating"
    generated_gui_config = os.path.join(
        plugin_dir, "gui_texture", "gui.config"
    )
    os.makedirs(os.path.dirname(generated_gui_config), exist_ok=True)
    with open(generated_gui_config, "w", encoding="utf-8") as file:
        file.write("\n".join(
            ET.tostring(element, encoding="unicode")
            for element in gui_config_root
        ))
    gui_config = generated_gui_config
    if headless:
        gazebo_command = (
            f"ign gazebo -s -r --render-engine-server {render_engine}"
        )
        gazebo_arguments = [gazebo_command, world_file]
    else:
        gazebo_command = (
            f"ign gazebo -r --render-engine {render_engine} "
            f"--render-engine-gui {render_engine}"
        )
        gazebo_arguments = [
            gazebo_command,
            world_file,
            "--gui-config",
            gui_config,
        ]
    ign_gazebo = ExecuteProcess(
        cmd=gazebo_arguments,
        output="screen",
        additional_env=env,
        shell=True,
    )
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )
    gazebo_spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-topic", "/robot_description",
            "-name", "crane_plus",
            "-z", "1.015",
            "-allow_renaming", "true",
        ],
    )
    spawn_joint_state_controller = ExecuteProcess(
        cmd=["ros2 run controller_manager spawner joint_state_broadcaster"],
        shell=True,
        output="screen",
    )
    spawn_arm_controller = ExecuteProcess(
        cmd=["ros2 run controller_manager spawner crane_plus_arm_controller"],
        shell=True,
        output="screen",
    )
    spawn_gripper_controller = ExecuteProcess(
        cmd=["ros2 run controller_manager spawner crane_plus_gripper_controller"],
        shell=True,
        output="screen",
    )
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
            "/paper/image@sensor_msgs/msg/Image[ignition.msgs.Image",
            "/paper/texture_update@sensor_msgs/msg/Image]ignition.msgs.Image",
        ],
        output="screen",
    )
    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_semantic,
            kinematics,
            ompl,
            moveit_controllers,
            {
                "trajectory_execution.allowed_execution_duration_scaling": 1.2,
                "trajectory_execution.allowed_goal_duration_margin": 0.5,
                "trajectory_execution.allowed_start_tolerance": 0.1,
                "publish_planning_scene": True,
                "publish_geometry_updates": True,
                "publish_state_updates": True,
                "publish_transforms_updates": True,
            },
        ],
    )

    attach_existing_gazebo = os.environ.get(
        "PAI_ATTACH_EXISTING_GAZEBO", ""
    ).lower() in {"1", "true", "yes", "on"}
    actions = [SetParameter(name="use_sim_time", value=True)]
    if not attach_existing_gazebo:
        actions.append(ign_gazebo)
    actions.append(robot_state_publisher)
    if not attach_existing_gazebo:
        actions.append(gazebo_spawn_entity)
    actions.append(move_group)
    if not attach_existing_gazebo:
        actions.extend([
            spawn_joint_state_controller,
            spawn_arm_controller,
            spawn_gripper_controller,
        ])
    actions.append(bridge)
    return LaunchDescription(actions)
