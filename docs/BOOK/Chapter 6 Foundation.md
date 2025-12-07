# Chapter 6: Gazebo Simulation

> "Simulation allows us to fail fast and learn faster. Test thousands of scenarios virtually before risking expensive hardware."

## Table of Contents

1. [Introduction to Gazebo](#introduction)
2. [Gazebo Architecture](#architecture)
3. [World Files and Environments](#worlds)
4. [Robot Models (URDF/SDF)](#models)
5. [Sensors in Simulation](#sensors)
6. [Physics Engine Configuration](#physics)
7. [ROS 2 Integration](#ros2-integration)
8. [Spawning and Controlling Robots](#spawning)
9. [Advanced Simulation Techniques](#advanced)
10. [Performance Optimization](#optimization)

---

## Introduction to Gazebo {#introduction}

**Gazebo** is a powerful 3D robot simulator that provides realistic physics, sensors, and rendering. It's the industry-standard tool for testing robots before deployment.

### Why Use Simulation?

**Benefits:**
- ✅ **Safe testing** - No risk to hardware or people
- ✅ **Cost-effective** - Test before building
- ✅ **Rapid iteration** - Modify and retest quickly
- ✅ **Extreme scenarios** - Test dangerous situations
- ✅ **Reproducibility** - Exact same conditions every time
- ✅ **Scalability** - Test multiple robots simultaneously

**Limitations:**
- ❌ Reality gap - Simulation isn't perfect
- ❌ Computational cost - Complex scenes need powerful hardware
- ❌ Sensor noise - Hard to model exactly
- ❌ Unexpected real-world factors

### Gazebo Versions

| Version | ROS 2 Compatibility | Status |
|---------|-------------------|--------|
| **Gazebo Classic (11)** | Foxy, Humble | Legacy |
| **Gazebo Fortress** | Humble | Stable |
| **Gazebo Garden** | Humble, Iron | Current |
| **Gazebo Harmonic** | Iron, Jazzy | Latest |

---

## Gazebo Architecture {#architecture}

### Core Components

```
┌─────────────────────────────────────┐
│         Gazebo Client (GUI)         │
│  ┌─────────┐  ┌──────────────────┐ │
│  │ Viewer  │  │  Plugin Widgets  │ │
│  └─────────┘  └──────────────────┘ │
└──────────────┬──────────────────────┘
               │ Communication
┌──────────────┴──────────────────────┐
│        Gazebo Server (gzserver)     │
│  ┌──────────┐  ┌────────────────┐  │
│  │ Physics  │  │   Rendering    │  │
│  │  Engine  │  │    Engine      │  │
│  └──────────┘  └────────────────┘  │
│  ┌──────────┐  ┌────────────────┐  │
│  │ Sensors  │  │    Plugins     │  │
│  └──────────┘  └────────────────┘  │
└─────────────────────────────────────┘
```

### Installation

```bash
# Gazebo Fortress (recommended for Humble)
sudo apt install ros-humble-gazebo-ros-pkgs

# Gazebo Garden
sudo apt install ros-humble-ros-gz

# Check installation
gz sim --version
```

### Launching Gazebo

```bash
# Empty world
gz sim empty.sdf

# With ROS 2 bridge
ros2 launch gazebo_ros gazebo.launch.py

# Specific world
gz sim worlds/shapes.sdf
```

---

## World Files and Environments {#worlds}

### SDF World Structure

**empty_world.sdf:**

```xml
<?xml version="1.0"?>
<sdf version="1.8">
  <world name="empty_world">
    
    <!-- Physics -->
    <physics name="default_physics" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
    
    <!-- Lighting -->
    <light name="sun" type="directional">
      <pose>0 0 10 0 0 0</pose>
      <diffuse>1 1 1 1</diffuse>
      <specular>0.5 0.5 0.5 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
      <attenuation>
        <range>1000</range>
      </attenuation>
    </light>
    
    <!-- Ground Plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
  </world>
</sdf>
```

### Creating Complex Environments

**warehouse_world.sdf:**

```xml
<sdf version="1.8">
  <world name="warehouse">
    
    <include>
      <uri>model://sun</uri>
    </include>
    
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <!-- Warehouse building -->
    <model name="warehouse_building">
      <pose>0 0 0 0 0 0</pose>
      <static>true</static>
      
      <!-- Walls -->
      <link name="wall_north">
        <pose>0 10 1.5 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>20 0.2 3</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>20 0.2 3</size>
            </box>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
          </material>
        </visual>
      </link>
      
      <!-- Add more walls... -->
    </model>
    
    <!-- Shelving units -->
    <include>
      <uri>model://shelf</uri>
      <name>shelf_1</name>
      <pose>2 2 0 0 0 0</pose>
    </include>
    
    <!-- Obstacles -->
    <include>
      <uri>model://cardboard_box</uri>
      <name>box_1</name>
      <pose>-3 -2 0.5 0 0 0.5</pose>
    </include>
    
  </world>
</sdf>
```

### Environment Best Practices

| Element | Recommendation |
|---------|---------------|
| **Lighting** | Use multiple light sources for realism |
| **Ground** | Add texture for visual odometry testing |
| **Obstacles** | Include various shapes and sizes |
| **Scale** | Match real-world dimensions |
| **Complexity** | Balance realism vs performance |

---

## Robot Models (URDF/SDF) {#models}

### URDF Basics

URDF (Unified Robot Description Format) describes robot kinematics and dynamics.

**simple_robot.urdf:**

```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  
  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    
    <collision>
      <geometry>
        <cylinder length="0.1" radius="0.2"/>
      </geometry>
    </collision>
    
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0"
               iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>
  
  <!-- Left Wheel -->
  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
    </collision>
    
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0"
               iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>
  
  <!-- Left Wheel Joint -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.25 0" rpy="1.57 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
  
  <!-- Right Wheel (similar structure) -->
  <!-- ... -->
  
  <!-- Caster Wheel -->
  <link name="caster">
    <visual>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
      <material name="grey">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    
    <collision>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
      <surface>
        <friction>
          <ode>
            <mu>0.0</mu>
            <mu2>0.0</mu2>
          </ode>
        </friction>
      </surface>
    </collision>
    
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.001" ixy="0" ixz="0"
               iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
  </link>
  
  <joint name="caster_joint" type="fixed">
    <parent link="base_link"/>
    <child link="caster"/>
    <origin xyz="-0.15 0 -0.05"/>
  </joint>
  
</robot>
```

### Adding Gazebo-Specific Tags

```xml
<robot name="my_robot">
  <!-- ... URDF content ... -->
  
  <!-- Gazebo materials -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
    <mu1>0.2</mu1>
    <mu2>0.2</mu2>
  </gazebo>
  
  <gazebo reference="left_wheel">
    <material>Gazebo/Black</material>
    <mu1>1.0</mu1>
    <mu2>1.0</mu2>
    <kp>1000000.0</kp>
    <kd>100.0</kd>
  </gazebo>
  
  <!-- Differential drive plugin -->
  <gazebo>
    <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
      <update_rate>50</update_rate>
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>0.5</wheel_separation>
      <wheel_diameter>0.2</wheel_diameter>
      <max_wheel_torque>20</max_wheel_torque>
      <command_topic>cmd_vel</command_topic>
      <odometry_topic>odom</odometry_topic>
      <odometry_frame>odom</odometry_frame>
      <robot_base_frame>base_link</robot_base_frame>
      <publish_odom>true</publish_odom>
      <publish_odom_tf>true</publish_odom_tf>
      <publish_wheel_tf>true</publish_wheel_tf>
    </plugin>
  </gazebo>
  
</robot>
```

### URDF Generation with xacro

**robot.urdf.xacro:**

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="my_robot">
  
  <!-- Parameters -->
  <xacro:property name="wheel_radius" value="0.1"/>
  <xacro:property name="wheel_length" value="0.05"/>
  <xacro:property name="wheel_separation" value="0.5"/>
  
  <!-- Macro for wheels -->
  <xacro:macro name="wheel" params="prefix reflect">
    <link name="${prefix}_wheel">
      <visual>
        <geometry>
          <cylinder length="${wheel_length}" radius="${wheel_radius}"/>
        </geometry>
        <material name="black"/>
      </visual>
      
      <collision>
        <geometry>
          <cylinder length="${wheel_length}" radius="${wheel_radius}"/>
        </geometry>
      </collision>
      
      <inertial>
        <mass value="0.5"/>
        <xacro:cylinder_inertia m="0.5" r="${wheel_radius}" h="${wheel_length}"/>
      </inertial>
    </link>
    
    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="base_link"/>
      <child link="${prefix}_wheel"/>
      <origin xyz="0 ${reflect * wheel_separation/2} 0" rpy="1.57 0 0"/>
      <axis xyz="0 0 1"/>
    </joint>
  </xacro:macro>
  
  <!-- Inertia macros -->
  <xacro:macro name="cylinder_inertia" params="m r h">
    <inertia ixx="${m*(3*r*r+h*h)/12}" ixy="0" ixz="0"
             iyy="${m*(3*r*r+h*h)/12}" iyz="0"
             izz="${m*r*r/2}"/>
  </xacro:macro>
  
  <!-- Base link -->
  <link name="base_link">
    <!-- ... -->
  </link>
  
  <!-- Instantiate wheels -->
  <xacro:wheel prefix="left" reflect="1"/>
  <xacro:wheel prefix="right" reflect="-1"/>
  
</robot>
```

**Process xacro:**

```bash
ros2 run xacro xacro robot.urdf.xacro > robot.urdf
```

---

## Sensors in Simulation {#sensors}

### Camera Sensor

```xml
<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <update_rate>30</update_rate>
    <camera name="camera">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>1920</width>
        <height>1080</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <ros>
        <namespace>/robot</namespace>
        <remapping>image_raw:=camera/image</remapping>
        <remapping>camera_info:=camera/camera_info</remapping>
      </ros>
      <camera_name>camera</camera_name>
      <frame_name>camera_link</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

### LiDAR Sensor

```xml
<gazebo reference="lidar_link">
  <sensor name="lidar" type="ray">
    <pose>0 0 0.1 0 0 0</pose>
    <update_rate>10</update_rate>
    <visualize>true</visualize>
    
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.10</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.01</stddev>
      </noise>
    </ray>
    
    <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>/robot</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
      <frame_name>lidar_link</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

### Depth Camera

```xml
<gazebo reference="depth_camera_link">
  <sensor name="depth_camera" type="depth">
    <update_rate>20</update_rate>
    <camera name="depth_camera">
      <horizontal_fov>1.047198</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.05</near>
        <far>10</far>
      </clip>
    </camera>
    
    <plugin name="depth_camera_controller" filename="libgazebo_ros_camera.so">
      <ros>
        <namespace>/robot</namespace>
      </ros>
      <camera_name>depth_camera</camera_name>
      <frame_name>depth_camera_link</frame_name>
      <min_depth>0.1</min_depth>
      <max_depth>10.0</max_depth>
    </plugin>
  </sensor>
</gazebo>
```

### IMU Sensor

```xml
<gazebo reference="imu_link">
  <sensor name="imu" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    
    <plugin name="imu_plugin" filename="libgazebo_ros_imu_sensor.so">
      <ros>
        <namespace>/robot</namespace>
        <remapping>~/out:=imu/data</remapping>
      </ros>
      <initial_orientation_as_reference>false</initial_orientation_as_reference>
      <frame_name>imu_link</frame_name>
    </plugin>
    
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </z>
      </angular_velocity>
      
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
  </sensor>
</gazebo>
```

---

## Physics Engine Configuration {#physics}

### Physics Parameters

```xml
<world name="my_world">
  <physics name="default_physics" type="ode">
    <max_step_size>0.001</max_step_size>
    <real_time_factor>1.0</real_time_factor>
    <real_time_update_rate>1000</real_time_update_rate>
    
    <!-- Gravity -->
    <gravity>0 0 -9.81</gravity>
    
    <!-- Solver parameters -->
    <ode>
      <solver>
        <type>quick</type>
        <iters>50</iters>
        <sor>1.3</sor>
      </solver>
      <constraints>
        <cfm>0.0</cfm>
        <erp>0.2</erp>
        <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
        <contact_surface_layer>0.001</contact_surface_layer>
      </constraints>
    </ode>
  </physics>
</world>
```

### Contact Properties

```xml
<gazebo reference="wheel_link">
  <mu1>1.0</mu1>  <!-- Friction coefficient 1 -->
  <mu2>1.0</mu2>  <!-- Friction coefficient 2 -->
  <kp>1e6</kp>    <!-- Contact stiffness -->
  <kd>1.0</kd>    <!-- Contact damping -->
  <minDepth>0.001</minDepth>
  <maxContacts>10</maxContacts>
</gazebo>
```

---

## ROS 2 Integration {#ros2-integration}

### Launch File with Gazebo

```python
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # Package directories
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_robot = get_package_share_directory('my_robot_description')
    
    # World file
    world_file = os.path.join(pkg_robot, 'worlds', 'warehouse.sdf')
    
    # URDF file
    urdf_file = os.path.join(pkg_robot, 'urdf', 'robot.urdf')
    
    # Gazebo server
    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')
        ]),
        launch_arguments={'world': world_file}.items()
    )
    
    # Gazebo client
    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')
        ])
    )
    
    # Spawn robot
    spawn_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'my_robot',
            '-file', urdf_file,
            '-x', '0',
            '-y', '0',
            '-z', '0.5'
        ],
        output='screen'
    )
    
    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': open(urdf_file).read()}],
        output='screen'
    )
    
    return LaunchDescription([
        gzserver,
        gzclient,
        robot_state_publisher,
        spawn_robot
    ])
```

---

## Spawning and Controlling Robots {#spawning}

### Spawn Entity Service

```bash
# Spawn from URDF file
ros2 run gazebo_ros spawn_entity.py \
  -entity my_robot \
  -file /path/to/robot.urdf \
  -x 0 -y 0 -z 0.5

# Spawn from topic
ros2 run gazebo_ros spawn_entity.py \
  -entity my_robot \
  -topic /robot_description \
  -x 1 -y 2 -z 0.3
```

### Control Robot

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.time = 0.0
    
    def control_loop(self):
        msg = Twist()
        
        # Circle motion
        msg.linear.x = 0.5
        msg.angular.z = 0.3
        
        self.publisher.publish(msg)
        self.time += 0.1

def main():
    rclpy.init()
    controller = RobotController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()
```

---

## Advanced Simulation Techniques {#advanced}

### Model States

```python
from gazebo_msgs.srv import GetModelState, SetModelState
from gazebo_msgs.msg import ModelState

class ModelStateManager(Node):
    def __init__(self):
        super().__init__('model_state_manager')
        
        self.get_state_client = self.create_client(
            GetModelState, '/gazebo/get_model_state')
        
        self.set_state_client = self.create_client(
            SetModelState, '/gazebo/set_model_state')
    
    def get_robot_pose(self):
        request = GetModelState.Request()
        request.model_name = 'my_robot'
        
        future = self.get_state_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        response = future.result()
        return response.pose
    
    def reset_robot(self):
        request = SetModelState.Request()
        request.model_state = ModelState()
        request.model_state.model_name = 'my_robot'
        request.model_state.pose.position.x = 0.0
        request.model_state.pose.position.y = 0.0
        request.model_state.pose.position.z = 0.5
        
        self.set_state_client.call_async(request)
```

### Custom Plugins

**Simple plugin in C++:**

```cpp
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>

namespace gazebo
{
  class CustomPlugin : public ModelPlugin
  {
  public:
    void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
    {
      this->model = _model;
      
      // Connect to update event
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
        std::bind(&CustomPlugin::OnUpdate, this));
      
      gzmsg << "Custom plugin loaded!" << std::endl;
    }
    
    void OnUpdate()
    {
      // Plugin logic here
      auto pose = this->model->WorldPose();
      // Do something with pose
    }
    
  private:
    physics::ModelPtr model;
    event::ConnectionPtr updateConnection;
  };
  
  GZ_REGISTER_MODEL_PLUGIN(CustomPlugin)
}
```

---

## Performance Optimization {#optimization}

### Optimization Strategies

| Technique | Description | Impact |
|-----------|-------------|--------|
| **Reduce mesh complexity** | Use simplified collision meshes | High |
| **Lower update rates** | Reduce sensor frequencies | Medium |
| **Adjust physics step** | Increase max_step_size | Medium |
| **Disable GUI** | Run headless with gzserver only | High |
| **Use simpler materials** | Avoid complex shaders | Low |
| **Limit sensors** | Only enable needed sensors | Medium |

### Headless Mode

```bash
# Run without GUI
ros2 launch my_robot_gazebo simulation.launch.py gui:=false
```

### Benchmark Performance

```python
class PerformanceMonitor(Node):
    def __init__(self):
        super().__init__('performance_monitor')
        
        self.create_timer(1.0, self.log_performance)
        self.last_time = self.get_clock().now()
        self.update_count = 0
    
    def log_performance(self):
        current_time = self.get_clock().now()
        duration = (current_time - self.last_time).nanoseconds / 1e9
        
        fps = self.update_count / duration
        self.get_logger().info(f'Simulation FPS: {fps:.2f}')
        
        self.last_time = current_time
        self.update_count = 0
```

---

## Summary

Gazebo simulation is essential for safe, cost-effective robot development. Master simulation before moving to real hardware, but always remember the reality gap exists.

**Key Takeaways:**

1. ✅ Use simulation for rapid prototyping and testing
2. ✅ Create realistic worlds with proper physics
3. ✅ Model sensors accurately with noise
4. ✅ Integrate tightly with ROS 2
5. ✅ Optimize for performance when needed
6. ✅ Always validate in real world eventually

**Next Chapter:**

Chapter 7 will explore URDF, SDF, and Unity for advanced robot modeling and visualization.

---

*This chapter introduced Gazebo simulation for robotics. You can now create virtual environments, model robots, and test algorithms safely before real-world deployment.*