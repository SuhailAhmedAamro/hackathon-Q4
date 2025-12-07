# Chapter 7: URDF, SDF & Unity Visualization

> "A robot is only as good as its model. Accurate representation is the foundation of simulation and control."

## Table of Contents

1. [Introduction to Robot Description](#introduction)
2. [URDF Deep Dive](#urdf)
3. [SDF Format](#sdf)
4. [Xacro for Modularity](#xacro)
5. [Unity for Robotics](#unity)
6. [ROS-Unity Integration](#ros-unity)
7. [Visualization Tools](#visualization)
8. [Best Practices](#best-practices)

---

## Introduction to Robot Description {#introduction}

Robot description formats define the physical and visual properties of robots for simulation and visualization.

### Format Comparison

| Format | Purpose | Strengths | Weaknesses |
|--------|---------|-----------|------------|
| **URDF** | ROS robot description | ROS integration, simple | Limited features, no sensors |
| **SDF** | Gazebo simulation | Full simulation support | More complex |
| **Xacro** | URDF templating | Reusable, parametric | Processing step needed |
| **Unity** | 3D visualization | Beautiful graphics | Heavier, game engine |

---

## URDF Deep Dive {#urdf}

### Complete URDF Structure

```xml
<?xml version="1.0"?>
<robot name="mobile_manipulator">
  
  <!-- Materials -->
  <material name="blue">
    <color rgba="0 0 0.8 1"/>
  </material>
  
  <material name="black">
    <color rgba="0 0 0 1"/>
  </material>
  
  <material name="white">
    <color rgba="1 1 1 1"/>
  </material>
  
  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.6 0.4 0.2"/>
      </geometry>
      <material name="blue"/>
    </visual>
    
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.6 0.4 0.2"/>
      </geometry>
    </collision>
    
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="15.0"/>
      <inertia ixx="0.175" ixy="0" ixz="0"
               iyy="0.495" iyz="0"
               izz="0.62"/>
    </inertial>
  </link>
  
  <!-- Left Wheel -->
  <link name="left_wheel">
    <visual>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <material name="black"/>
    </visual>
    
    <collision>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
    </collision>
    
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.00277" ixy="0" ixz="0"
               iyy="0.00277" iyz="0"
               izz="0.005"/>
    </inertial>
  </link>
  
  <!-- Left Wheel Joint -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0.2 0.25 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <dynamics damping="0.1" friction="0.1"/>
  </joint>
  
  <!-- Right Wheel -->
  <link name="right_wheel">
    <visual>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <material name="black"/>
    </visual>
    
    <collision>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
    </collision>
    
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.00277" ixy="0" ixz="0"
               iyy="0.00277" iyz="0"
               izz="0.005"/>
    </inertial>
  </link>
  
  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0.2 -0.25 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <dynamics damping="0.1" friction="0.1"/>
  </joint>
  
  <!-- Caster Wheel -->
  <link name="caster">
    <visual>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
      <material name="white"/>
    </visual>
    
    <collision>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
    </collision>
    
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.0005" ixy="0" ixz="0"
               iyy="0.0005" iyz="0"
               izz="0.0005"/>
    </inertial>
  </link>
  
  <joint name="caster_joint" type="fixed">
    <parent link="base_link"/>
    <child link="caster"/>
    <origin xyz="-0.2 0 -0.05"/>
  </joint>
  
  <!-- Arm Links -->
  <link name="arm_base">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
      <material name="blue"/>
    </visual>
    
    <collision>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
    </collision>
    
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0"
               iyy="0.001" iyz="0"
               izz="0.001"/>
    </inertial>
  </link>
  
  <joint name="arm_base_joint" type="revolute">
    <parent link="base_link"/>
    <child link="arm_base"/>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="1.0"/>
  </joint>
  
  <!-- Add more arm links... -->
  
</robot>
```

### Joint Types

| Type | DOF | Description | Use Case |
|------|-----|-------------|----------|
| **fixed** | 0 | No movement | Sensors, static attachments |
| **revolute** | 1 | Rotation with limits | Robot joints |
| **continuous** | 1 | Unlimited rotation | Wheels |
| **prismatic** | 1 | Linear motion | Linear actuators |
| **planar** | 2 | Plane motion | Mobile platforms |
| **floating** | 6 | Free motion | Underwater robots |

### Inertia Calculations

```python
# Inertia calculator utilities
import math

def box_inertia(mass, x, y, z):
    """Calculate inertia for box"""
    ixx = (mass / 12.0) * (y**2 + z**2)
    iyy = (mass / 12.0) * (x**2 + z**2)
    izz = (mass / 12.0) * (x**2 + y**2)
    return {'ixx': ixx, 'iyy': iyy, 'izz': izz}

def cylinder_inertia(mass, radius, length):
    """Calculate inertia for cylinder"""
    ixx = (mass / 12.0) * (3 * radius**2 + length**2)
    iyy = ixx
    izz = 0.5 * mass * radius**2
    return {'ixx': ixx, 'iyy': iyy, 'izz': izz}

def sphere_inertia(mass, radius):
    """Calculate inertia for sphere"""
    i = 0.4 * mass * radius**2
    return {'ixx': i, 'iyy': i, 'izz': i}

# Example usage
box = box_inertia(15.0, 0.6, 0.4, 0.2)
print(f"Box inertia: ixx={box['ixx']:.3f}, iyy={box['iyy']:.3f}, izz={box['izz']:.3f}")
```

---

## SDF Format {#sdf}

### SDF vs URDF

```xml
<?xml version="1.0"?>
<sdf version="1.8">
  <model name="mobile_robot">
    
    <!-- Model pose -->
    <pose>0 0 0.1 0 0 0</pose>
    
    <!-- Links -->
    <link name="base_link">
      <pose>0 0 0 0 0 0</pose>
      
      <inertial>
        <mass>15.0</mass>
        <inertia>
          <ixx>0.175</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.495</iyy>
          <iyz>0</iyz>
          <izz>0.62</izz>
        </inertia>
      </inertial>
      
      <visual name="base_visual">
        <geometry>
          <box>
            <size>0.6 0.4 0.2</size>
          </box>
        </geometry>
        <material>
          <ambient>0 0 0.8 1</ambient>
          <diffuse>0 0 0.8 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>
      
      <collision name="base_collision">
        <geometry>
          <box>
            <size>0.6 0.4 0.2</size>
          </box>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>0.6</mu>
              <mu2>0.6</mu2>
            </ode>
          </friction>
        </surface>
      </collision>
      
      <!-- Sensors in SDF -->
      <sensor name="imu_sensor" type="imu">
        <always_on>true</always_on>
        <update_rate>100</update_rate>
        <pose>0 0 0 0 0 0</pose>
      </sensor>
      
      <sensor name="camera" type="camera">
        <pose>0.3 0 0.2 0 0 0</pose>
        <camera>
          <horizontal_fov>1.047</horizontal_fov>
          <image>
            <width>1920</width>
            <height>1080</height>
          </image>
          <clip>
            <near>0.1</near>
            <far>100</far>
          </clip>
        </camera>
        <always_on>true</always_on>
        <update_rate>30</update_rate>
      </sensor>
    </link>
    
    <!-- Joints -->
    <joint name="left_wheel_joint" type="revolute">
      <parent>base_link</parent>
      <child>left_wheel</child>
      <pose>0 0 0 0 0 0</pose>
      <axis>
        <xyz>0 1 0</xyz>
        <dynamics>
          <damping>0.1</damping>
          <friction>0.1</friction>
        </dynamics>
      </axis>
    </joint>
    
    <!-- Plugins -->
    <plugin name="differential_drive_controller" 
            filename="libgazebo_ros_diff_drive.so">
      <update_rate>50</update_rate>
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>0.5</wheel_separation>
      <wheel_diameter>0.2</wheel_diameter>
      <max_wheel_torque>20</max_wheel_torque>
      <command_topic>cmd_vel</command_topic>
      <odometry_topic>odom</odometry_topic>
    </plugin>
    
  </model>
</sdf>
```

---

## Xacro for Modularity {#xacro}

### Xacro Properties

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="modular_robot">
  
  <!-- Properties -->
  <xacro:property name="wheel_radius" value="0.1"/>
  <xacro:property name="wheel_width" value="0.05"/>
  <xacro:property name="wheel_mass" value="1.0"/>
  <xacro:property name="wheel_separation" value="0.5"/>
  
  <xacro:property name="base_length" value="0.6"/>
  <xacro:property name="base_width" value="0.4"/>
  <xacro:property name="base_height" value="0.2"/>
  <xacro:property name="base_mass" value="15.0"/>
  
  <!-- PI constant -->
  <xacro:property name="pi" value="3.14159265359"/>
  
  <!-- Calculated properties -->
  <xacro:property name="wheel_offset_y" value="${wheel_separation/2}"/>
  
</robot>
```

### Xacro Macros

```xml
<!-- Wheel macro -->
<xacro:macro name="wheel" params="prefix reflect">
  
  <link name="${prefix}_wheel">
    <visual>
      <origin xyz="0 0 0" rpy="${pi/2} 0 0"/>
      <geometry>
        <cylinder length="${wheel_width}" radius="${wheel_radius}"/>
      </geometry>
      <material name="black"/>
    </visual>
    
    <collision>
      <origin xyz="0 0 0" rpy="${pi/2} 0 0"/>
      <geometry>
        <cylinder length="${wheel_width}" radius="${wheel_radius}"/>
      </geometry>
    </collision>
    
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="${wheel_mass}"/>
      <xacro:cylinder_inertia m="${wheel_mass}" 
                              r="${wheel_radius}" 
                              h="${wheel_width}"/>
    </inertial>
  </link>
  
  <joint name="${prefix}_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="${prefix}_wheel"/>
    <origin xyz="0.2 ${reflect * wheel_offset_y} 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <dynamics damping="0.1" friction="0.1"/>
  </joint>
  
  <gazebo reference="${prefix}_wheel">
    <mu1>1.0</mu1>
    <mu2>1.0</mu2>
    <kp>10000000.0</kp>
    <kd>1.0</kd>
    <material>Gazebo/Black</material>
  </gazebo>
  
</xacro:macro>

<!-- Inertia macros -->
<xacro:macro name="cylinder_inertia" params="m r h">
  <inertia ixx="${m*(3*r*r+h*h)/12}" ixy="0" ixz="0"
           iyy="${m*(3*r*r+h*h)/12}" iyz="0"
           izz="${m*r*r/2}"/>
</xacro:macro>

<xacro:macro name="box_inertia" params="m x y z">
  <inertia ixx="${m*(y*y+z*z)/12}" ixy="0" ixz="0"
           iyy="${m*(x*x+z*z)/12}" iyz="0"
           izz="${m*(x*x+y*y)/12}"/>
</xacro:macro>

<xacro:macro name="sphere_inertia" params="m r">
  <inertia ixx="${2*m*r*r/5}" ixy="0" ixz="0"
           iyy="${2*m*r*r/5}" iyz="0"
           izz="${2*m*r*r/5}"/>
</xacro:macro>

<!-- Use macros -->
<xacro:wheel prefix="left" reflect="1"/>
<xacro:wheel prefix="right" reflect="-1"/>
```

### Including External Files

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="complete_robot">
  
  <!-- Include common properties -->
  <xacro:include filename="$(find my_robot_description)/urdf/common_properties.xacro"/>
  
  <!-- Include wheel macro -->
  <xacro:include filename="$(find my_robot_description)/urdf/wheel.xacro"/>
  
  <!-- Include sensor macros -->
  <xacro:include filename="$(find my_robot_description)/urdf/sensors.xacro"/>
  
  <!-- Use included macros -->
  <xacro:wheel prefix="left" reflect="1"/>
  <xacro:wheel prefix="right" reflect="-1"/>
  
  <xacro:lidar_sensor parent="base_link" xyz="0.3 0 0.2"/>
  <xacro:camera_sensor parent="base_link" xyz="0.3 0 0.25"/>
  
</robot>
```

### Processing Xacro Files

```bash
# Convert xacro to URDF
ros2 run xacro xacro robot.urdf.xacro > robot.urdf

# With parameters
ros2 run xacro xacro robot.urdf.xacro \
  wheel_radius:=0.15 \
  wheel_separation:=0.6 \
  > robot_custom.urdf

# Check URDF validity
check_urdf robot.urdf

# Visualize URDF tree
urdf_to_graphiz robot.urdf
```

---

## Unity for Robotics {#unity}

### Unity Robotics Hub

Unity provides photorealistic visualization and simulation capabilities.

**Installation:**

```bash
# Install Unity Hub
# Download from unity.com

# Install Unity Robotics packages via Package Manager:
# - ROS TCP Connector
# - URDF Importer
# - Robotics Visualizations
```

### URDF Import to Unity

**Unity URDF Importer Settings:**

```csharp
// RobotImporter.cs
using UnityEngine;
using Unity.Robotics.UrdfImporter;

public class RobotImporter : MonoBehaviour
{
    public string urdfPath = "Assets/URDF/robot.urdf";
    
    void Start()
    {
        // Import URDF
        GameObject robot = UrdfRobotExtensions.CreateRuntime(urdfPath);
        
        // Configure physics
        ConfigurePhysics(robot);
        
        // Add visualization components
        AddVisualization(robot);
    }
    
    void ConfigurePhysics(GameObject robot)
    {
        // Set up articulation bodies
        foreach (var joint in robot.GetComponentsInChildren<ArticulationBody>())
        {
            joint.solverIterations = 20;
            joint.solverVelocityIterations = 20;
        }
    }
    
    void AddVisualization(GameObject robot)
    {
        // Add custom shaders, effects
        var renderer = robot.GetComponentInChildren<Renderer>();
        if (renderer != null)
        {
            renderer.material.shader = Shader.Find("Standard");
        }
    }
}
```

### Unity Graphics Settings

```json
{
  "render_pipeline": "URP",
  "quality": {
    "anti_aliasing": "4x MSAA",
    "shadows": "Hard Shadows",
    "texture_quality": "Full Res",
    "vsync": true,
    "target_fps": 60
  },
  "post_processing": {
    "ambient_occlusion": true,
    "bloom": true,
    "color_grading": true,
    "depth_of_field": false
  }
}
```

---

## ROS-Unity Integration {#ros-unity}

### ROS TCP Endpoint

**Unity → ROS 2 Communication:**

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;

public class RobotController : MonoBehaviour
{
    private ROSConnection ros;
    private string topicName = "/cmd_vel";
    
    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<TwistMsg>(topicName);
    }
    
    void Update()
    {
        // Get input
        float linear = Input.GetAxis("Vertical");
        float angular = Input.GetAxis("Horizontal");
        
        // Create message
        TwistMsg twist = new TwistMsg
        {
            linear = new Vector3Msg { x = linear, y = 0, z = 0 },
            angular = new Vector3Msg { x = 0, y = 0, z = angular }
        };
        
        // Publish
        ros.Publish(topicName, twist);
    }
}
```

**ROS 2 → Unity (Subscriber):**

```csharp
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class SensorVisualizer : MonoBehaviour
{
    private ROSConnection ros;
    
    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Subscribe<LaserScanMsg>("/scan", OnScanReceived);
    }
    
    void OnScanReceived(LaserScanMsg scan)
    {
        // Visualize LiDAR data
        for (int i = 0; i < scan.ranges.Length; i++)
        {
            float angle = scan.angle_min + i * scan.angle_increment;
            float range = scan.ranges[i];
            
            Vector3 point = new Vector3(
                range * Mathf.Cos(angle),
                0,
                range * Mathf.Sin(angle)
            );
            
            Debug.DrawLine(Vector3.zero, point, Color.red, 0.1f);
        }
    }
}
```

### Launch ROS TCP Endpoint

```python
# ros_tcp_endpoint.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ros_tcp_endpoint',
            executable='default_server_endpoint',
            parameters=[{
                'ROS_IP': '192.168.1.100',
                'ROS_TCP_PORT': 10000
            }],
            output='screen'
        )
    ])
```

---

## Visualization Tools {#visualization}

### RViz2

```bash
# Launch RViz2
rviz2

# With config file
rviz2 -d robot.rviz

# From launch file
Node(
    package='rviz2',
    executable='rviz2',
    arguments=['-d', rviz_config_path],
    output='screen'
)
```

**RViz Config Example:**

```yaml
Panels:
  - Class: rviz_common/Displays
  - Class: rviz_common/Views
  - Class: rviz_common/Time

Visualization Manager:
  Displays:
    - Class: rviz_default_plugins/RobotModel
      Robot Description: robot_description
      Visual Enabled: true
      
    - Class: rviz_default_plugins/TF
      Frames:
        All Enabled: true
      
    - Class: rviz_default_plugins/LaserScan
      Topic: /scan
      Size: 0.05
      Color: 255; 0; 0
      
    - Class: rviz_default_plugins/Camera
      Topic: /camera/image_raw
```

### Joint State Publisher GUI

```bash
# Install
sudo apt install ros-humble-joint-state-publisher-gui

# Launch
ros2 run joint_state_publisher_gui joint_state_publisher_gui
```

**Launch with robot:**

```python
Node(
    package='joint_state_publisher_gui',
    executable='joint_state_publisher_gui',
    output='screen'
),
Node(
    package='robot_state_publisher',
    executable='robot_state_publisher',
    parameters=[{'robot_description': robot_description}]
)
```

---

## Best Practices {#best-practices}

### URDF Best Practices

**✅ DO:**
- Use realistic mass and inertia values
- Add collision meshes separate from visual
- Use xacro for modularity
- Define materials consistently
- Include proper joint limits

**❌ DON'T:**
- Use zero or very small masses
- Ignore collision geometry
- Hardcode values (use properties)
- Forget to define axes
- Use complex meshes for collision

### File Organization

```
robot_description/
├── urdf/
│   ├── robot.urdf.xacro          # Main file
│   ├── common_properties.xacro   # Shared properties
│   ├── wheel.xacro               # Wheel macro
│   ├── sensors.xacro             # Sensor macros
│   └── gazebo.xacro              # Gazebo-specific
├── meshes/
│   ├── visual/                   # High-poly meshes
│   │   ├── base_link.dae
│   │   └── arm_link.dae
│   └── collision/                # Low-poly meshes
│       ├── base_link.stl
│       └── arm_link.stl
├── config/
│   └── robot.rviz
└── launch/
    └── display.launch.py
```

### Performance Tips

| Aspect | Recommendation |
|--------|---------------|
| **Mesh complexity** | Use simplified collision meshes |
| **Material count** | Minimize unique materials |
| **Joint count** | Only model necessary DOFs |
| **Sensor rate** | Match actual hardware rates |
| **Update frequency** | Balance accuracy vs performance |

---

## Summary

Robot description formats are the foundation of simulation and visualization. Master URDF/SDF for accurate robot modeling, use xacro for maintainability, and leverage Unity for beautiful visualization.

**Key Takeaways:**

1. ✅ URDF for ROS integration
2. ✅ SDF for Gazebo simulation
3. ✅ Xacro for modular, reusable descriptions
4. ✅ Unity for photorealistic visualization
5. ✅ Proper inertia and physics properties are critical
6. ✅ Organize models for maintainability

**Next Chapter:**

Chapter 8 will introduce NVIDIA Isaac Sim, a cutting-edge photorealistic simulation platform with GPU acceleration.

---

*This chapter covered robot description formats and visualization. You can now create accurate robot models for simulation and beautiful visualizations for presentation.*