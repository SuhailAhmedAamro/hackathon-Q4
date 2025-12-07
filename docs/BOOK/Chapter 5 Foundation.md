# Chapter 5: Building ROS 2 Packages

> "A well-organized package is the foundation of maintainable, scalable robotics software."

## Table of Contents

1. [Introduction to ROS 2 Packages](#introduction)
2. [Package Structure and Organization](#structure)
3. [Creating Python Packages](#python-packages)
4. [Creating C++ Packages](#cpp-packages)
5. [Launch Files](#launch-files)
6. [Configuration Management](#configuration)
7. [Building and Testing](#building)
8. [Package Dependencies](#dependencies)
9. [Best Practices](#best-practices)
10. [Complete Example Project](#example-project)

---

## Introduction to ROS 2 Packages {#introduction}

A **ROS 2 package** is the fundamental unit of software organization in ROS. Packages contain nodes, libraries, configuration files, and other resources needed for a specific functionality.

### What is a Package?

A package is a directory containing:

- **Source code** (nodes, libraries)
- **Configuration files** (parameters, launch files)
- **Resource files** (models, maps, data)
- **Build instructions** (package.xml, CMakeLists.txt or setup.py)
- **Documentation** (README, docs)

### Package Types

| Type | Build System | Languages | Use Case |
|------|--------------|-----------|----------|
| **ament_python** | setuptools | Python | Python-only nodes |
| **ament_cmake** | CMake | C++, Python | C++ nodes, mixed projects |
| **ament_cmake_python** | CMake + setuptools | C++, Python | Hybrid packages |

### Workspace Structure

```
ros2_workspace/
├── src/                      # Source space
│   ├── my_robot_pkg/
│   ├── my_sensors_pkg/
│   └── my_control_pkg/
├── build/                    # Build artifacts (generated)
├── install/                  # Installed packages (generated)
└── log/                      # Build logs (generated)
```

---

## Package Structure and Organization {#structure}

### Python Package Structure

```
my_robot_package/
├── my_robot_package/         # Python module
│   ├── __init__.py
│   ├── node1.py
│   ├── node2.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── launch/                   # Launch files
│   ├── robot.launch.py
│   └── simulation.launch.py
├── config/                   # Configuration files
│   ├── params.yaml
│   └── rviz_config.rviz
├── resource/                 # Resource marker
│   └── my_robot_package
├── test/                     # Unit tests
│   ├── test_node1.py
│   └── test_utils.py
├── package.xml               # Package metadata
├── setup.py                  # Build configuration
├── setup.cfg                 # Setup configuration
└── README.md                 # Documentation
```

### C++ Package Structure

```
my_cpp_package/
├── include/                  # Header files
│   └── my_cpp_package/
│       ├── node1.hpp
│       └── utilities.hpp
├── src/                      # Source files
│   ├── node1.cpp
│   ├── node2.cpp
│   └── utilities.cpp
├── launch/                   # Launch files
│   └── nodes.launch.py
├── config/                   # Configuration
│   └── params.yaml
├── test/                     # Unit tests
│   └── test_node1.cpp
├── CMakeLists.txt           # CMake build config
├── package.xml              # Package metadata
└── README.md
```

---

## Creating Python Packages {#python-packages}

### Step 1: Create Package

```bash
cd ~/ros2_ws/src
ros2 pkg create my_robot_controller \
  --build-type ament_python \
  --dependencies rclpy std_msgs geometry_msgs sensor_msgs
```

### Step 2: Understand Generated Files

**package.xml** - Package metadata:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" 
            schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_controller</name>
  <version>0.0.1</version>
  <description>Robot controller package</description>
  <maintainer email="you@example.com">Your Name</maintainer>
  <license>Apache License 2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>sensor_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

**setup.py** - Build configuration:

```python
from setuptools import setup

package_name = 'my_robot_controller'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/robot.launch.py']),
        ('share/' + package_name + '/config', ['config/params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='Robot controller package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'velocity_publisher = my_robot_controller.velocity_publisher:main',
            'position_tracker = my_robot_controller.position_tracker:main',
            'safety_monitor = my_robot_controller.safety_monitor:main',
        ],
    },
)
```

### Step 3: Create Nodes

**my_robot_controller/velocity_publisher.py:**

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class VelocityPublisher(Node):
    def __init__(self):
        super().__init__('velocity_publisher')
        
        # Declare parameters
        self.declare_parameter('linear_velocity', 0.5)
        self.declare_parameter('angular_velocity', 0.2)
        self.declare_parameter('publish_rate', 10.0)
        
        # Get parameters
        self.linear_vel = self.get_parameter('linear_velocity').value
        self.angular_vel = self.get_parameter('angular_velocity').value
        rate = self.get_parameter('publish_rate').value
        
        # Create publisher
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Create timer
        self.timer = self.create_timer(1.0 / rate, self.publish_velocity)
        
        self.get_logger().info(f'Velocity Publisher started')
        self.get_logger().info(f'Linear: {self.linear_vel}, Angular: {self.angular_vel}')
    
    def publish_velocity(self):
        msg = Twist()
        msg.linear.x = self.linear_vel
        msg.angular.z = self.angular_vel
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = VelocityPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**my_robot_controller/position_tracker.py:**

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
import math

class PositionTracker(Node):
    def __init__(self):
        super().__init__('position_tracker')
        
        # Subscribe to odometry
        self.subscription = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10)
        
        # Publish position
        self.position_pub = self.create_publisher(Point, 'current_position', 10)
        
        # Track total distance
        self.last_position = None
        self.total_distance = 0.0
        
        self.get_logger().info('Position Tracker started')
    
    def odom_callback(self, msg):
        current_pos = msg.pose.pose.position
        
        # Calculate distance traveled
        if self.last_position is not None:
            dx = current_pos.x - self.last_position.x
            dy = current_pos.y - self.last_position.y
            distance = math.sqrt(dx*dx + dy*dy)
            self.total_distance += distance
        
        self.last_position = current_pos
        
        # Publish current position
        position_msg = Point()
        position_msg.x = current_pos.x
        position_msg.y = current_pos.y
        position_msg.z = self.total_distance
        self.position_pub.publish(position_msg)
        
        self.get_logger().info(
            f'Position: ({current_pos.x:.2f}, {current_pos.y:.2f}), '
            f'Distance: {self.total_distance:.2f}m',
            throttle_duration_sec=1.0)  # Log once per second

def main(args=None):
    rclpy.init(args=args)
    node = PositionTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 4: Add Utilities Module

**my_robot_controller/utils/__init__.py:**

```python
from .math_utils import *
from .transform_utils import *
```

**my_robot_controller/utils/math_utils.py:**

```python
import math

def normalize_angle(angle):
    """Normalize angle to [-pi, pi]"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_heading(x1, y1, x2, y2):
    """Calculate heading angle from point1 to point2"""
    return math.atan2(y2 - y1, x2 - x1)
```

---

## Creating C++ Packages {#cpp-packages}

### Step 1: Create Package

```bash
ros2 pkg create my_cpp_controller \
  --build-type ament_cmake \
  --dependencies rclcpp std_msgs geometry_msgs sensor_msgs
```

### Step 2: Create Node

**src/velocity_publisher.cpp:**

```cpp
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>

class VelocityPublisher : public rclcpp::Node
{
public:
  VelocityPublisher() : Node("velocity_publisher")
  {
    // Declare parameters
    this->declare_parameter("linear_velocity", 0.5);
    this->declare_parameter("angular_velocity", 0.2);
    this->declare_parameter("publish_rate", 10.0);
    
    // Get parameters
    linear_vel_ = this->get_parameter("linear_velocity").as_double();
    angular_vel_ = this->get_parameter("angular_velocity").as_double();
    double rate = this->get_parameter("publish_rate").as_double();
    
    // Create publisher
    publisher_ = this->create_publisher<geometry_msgs::msg::Twist>(
      "cmd_vel", 10);
    
    // Create timer
    auto period = std::chrono::duration<double>(1.0 / rate);
    timer_ = this->create_wall_timer(
      period,
      std::bind(&VelocityPublisher::publishVelocity, this));
    
    RCLCPP_INFO(this->get_logger(), "Velocity Publisher started");
  }

private:
  void publishVelocity()
  {
    auto message = geometry_msgs::msg::Twist();
    message.linear.x = linear_vel_;
    message.angular.z = angular_vel_;
    publisher_->publish(message);
  }

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  double linear_vel_;
  double angular_vel_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<VelocityPublisher>());
  rclcpp::shutdown();
  return 0;
}
```

### Step 3: Configure CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.8)
project(my_cpp_controller)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Include directories
include_directories(include)

# Add executables
add_executable(velocity_publisher src/velocity_publisher.cpp)
ament_target_dependencies(velocity_publisher
  rclcpp
  geometry_msgs
)

add_executable(position_tracker src/position_tracker.cpp)
ament_target_dependencies(position_tracker
  rclcpp
  nav_msgs
  geometry_msgs
)

# Install executables
install(TARGETS
  velocity_publisher
  position_tracker
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
```

---

## Launch Files {#launch-files}

Launch files start multiple nodes with configuration.

### Python Launch File

**launch/robot.launch.py:**

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('my_robot_controller')
    
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time')
    
    declare_params_file = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg_dir, 'config', 'params.yaml'),
        description='Full path to params file')
    
    # Nodes
    velocity_publisher = Node(
        package='my_robot_controller',
        executable='velocity_publisher',
        name='velocity_publisher',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        output='screen'
    )
    
    position_tracker = Node(
        package='my_robot_controller',
        executable='position_tracker',
        name='position_tracker',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
        remappings=[
            ('/odom', '/robot/odom')
        ]
    )
    
    safety_monitor = Node(
        package='my_robot_controller',
        executable='safety_monitor',
        name='safety_monitor',
        output='screen'
    )
    
    return LaunchDescription([
        declare_use_sim_time,
        declare_params_file,
        velocity_publisher,
        position_tracker,
        safety_monitor
    ])
```

### Advanced Launch File with Conditions

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    GroupAction
)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch arguments
    use_rviz = LaunchConfiguration('use_rviz')
    use_gazebo = LaunchConfiguration('use_gazebo')
    robot_name = LaunchConfiguration('robot_name')
    
    # Declare arguments
    declare_use_rviz = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Start RViz')
    
    declare_use_gazebo = DeclareLaunchArgument(
        'use_gazebo',
        default_value='false',
        description='Start Gazebo')
    
    declare_robot_name = DeclareLaunchArgument(
        'robot_name',
        default_value='my_robot',
        description='Robot name')
    
    # RViz (conditional)
    rviz_config = PathJoinSubstitution([
        FindPackageShare('my_robot_controller'),
        'config',
        'robot.rviz'
    ])
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        condition=IfCondition(use_rviz),
        output='screen'
    )
    
    # Gazebo (conditional)
    gazebo_launch = IncludeLaunchDescription(
        PathJoinSubstitution([
            FindPackageShare('gazebo_ros'),
            'launch',
            'gazebo.launch.py'
        ]),
        condition=IfCondition(use_gazebo)
    )
    
    # Robot nodes (grouped with namespace)
    robot_group = GroupAction([
        Node(
            package='my_robot_controller',
            executable='velocity_publisher',
            namespace=robot_name,
            name='velocity_publisher',
            output='screen'
        ),
        Node(
            package='my_robot_controller',
            executable='position_tracker',
            namespace=robot_name,
            name='position_tracker',
            output='screen'
        )
    ])
    
    return LaunchDescription([
        declare_use_rviz,
        declare_use_gazebo,
        declare_robot_name,
        rviz_node,
        gazebo_launch,
        robot_group
    ])
```

### Running Launch Files

```bash
# Basic launch
ros2 launch my_robot_controller robot.launch.py

# With arguments
ros2 launch my_robot_controller robot.launch.py use_sim_time:=true

# With custom params
ros2 launch my_robot_controller robot.launch.py \
  params_file:=/path/to/custom_params.yaml
```

---

## Configuration Management {#configuration}

### YAML Parameters File

**config/params.yaml:**

```yaml
/**:
  ros__parameters:
    use_sim_time: false

velocity_publisher:
  ros__parameters:
    linear_velocity: 0.5
    angular_velocity: 0.2
    publish_rate: 10.0

position_tracker:
  ros__parameters:
    max_distance: 100.0
    log_rate: 1.0

safety_monitor:
  ros__parameters:
    min_distance: 0.3
    warning_distance: 0.5
    max_speed: 1.0
```

### Loading Parameters

```python
# In node constructor
self.declare_parameter('linear_velocity', 0.5)
linear_vel = self.get_parameter('linear_velocity').value

# From launch file
Node(
    package='my_robot_controller',
    executable='velocity_publisher',
    parameters=['/path/to/params.yaml']
)

# From command line
ros2 run my_robot_controller velocity_publisher \
  --ros-args -p linear_velocity:=1.0
```

### Dynamic Reconfigure

```python
from rcl_interfaces.msg import ParameterDescriptor

class DynamicNode(Node):
    def __init__(self):
        super().__init__('dynamic_node')
        
        # Declare parameter with descriptor
        descriptor = ParameterDescriptor(
            description='Maximum velocity',
            type=ParameterType.PARAMETER_DOUBLE,
            read_only=False
        )
        self.declare_parameter('max_velocity', 1.0, descriptor)
        
        # Add callback for parameter changes
        self.add_on_set_parameters_callback(self.parameter_callback)
    
    def parameter_callback(self, params):
        for param in params:
            if param.name == 'max_velocity':
                self.get_logger().info(f'Max velocity changed to {param.value}')
        return SetParametersResult(successful=True)
```

---

## Building and Testing {#building}

### Building Packages

```bash
# Build all packages
cd ~/ros2_ws
colcon build

# Build specific package
colcon build --packages-select my_robot_controller

# Build with debug symbols
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Debug

# Build with specific Python interpreter
colcon build --cmake-args -DPYTHON_EXECUTABLE=/usr/bin/python3.10

# Parallel build
colcon build --parallel-workers 4
```

### Testing

**Python Test (test/test_velocity_publisher.py):**

```python
import pytest
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from my_robot_controller.velocity_publisher import VelocityPublisher

def test_velocity_publisher():
    rclpy.init()
    
    # Create node
    node = VelocityPublisher()
    
    # Test parameters
    assert node.get_parameter('linear_velocity').value == 0.5
    assert node.get_parameter('angular_velocity').value == 0.2
    
    # Cleanup
    node.destroy_node()
    rclpy.shutdown()

def test_message_format():
    rclpy.init()
    
    received_msgs = []
    
    class TestSubscriber(Node):
        def __init__(self):
            super().__init__('test_subscriber')
            self.subscription = self.create_subscription(
                Twist,
                'cmd_vel',
                lambda msg: received_msgs.append(msg),
                10)
    
    publisher = VelocityPublisher()
    subscriber = TestSubscriber()
    
    # Spin for a bit
    for _ in range(10):
        rclpy.spin_once(publisher, timeout_sec=0.1)
        rclpy.spin_once(subscriber, timeout_sec=0.1)
    
    # Check received messages
    assert len(received_msgs) > 0
    assert received_msgs[0].linear.x == 0.5
    
    # Cleanup
    publisher.destroy_node()
    subscriber.destroy_node()
    rclpy.shutdown()
```

**Run Tests:**

```bash
# Run all tests
colcon test

# Run tests for specific package
colcon test --packages-select my_robot_controller

# Show test results
colcon test-result --verbose
```

---

## Package Dependencies {#dependencies}

### Types of Dependencies

```xml
<!-- Build dependency -->
<build_depend>rclcpp</build_depend>

<!-- Execution dependency -->
<exec_depend>python3-numpy</exec_depend>

<!-- Both build and exec -->
<depend>geometry_msgs</depend>

<!-- Test dependency -->
<test_depend>ament_cmake_pytest</test_depend>

<!-- Build tool -->
<buildtool_depend>ament_cmake</buildtool_depend>
```

### Managing Dependencies

```bash
# Install dependencies
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y

# Update rosdep
rosdep update
```

---

## Best Practices {#best-practices}

### Package Organization

**✅ DO:**
- Keep packages focused and modular
- Use meaningful package names
- Separate interface packages
- Include documentation
- Version your packages

**❌ DON'T:**
- Create monolithic packages
- Mix languages unnecessarily
- Hardcode paths
- Ignore dependencies
- Skip documentation

### Code Quality

```python
# Good: Clean, documented node
class WellDesignedNode(Node):
    """
    A well-designed ROS 2 node.
    
    Publishes: /output (std_msgs/String)
    Subscribes: /input (std_msgs/String)
    Parameters: rate (double)
    """
    
    def __init__(self):
        super().__init__('well_designed_node')
        
        # Parameters
        self.declare_parameter('rate', 10.0)
        rate = self.get_parameter('rate').value
        
        # Publishers
        self.publisher = self.create_publisher(String, 'output', 10)
        
        # Subscribers
        self.subscription = self.create_subscription(
            String, 'input', self.callback, 10)
        
        # Timers
        self.timer = self.create_timer(1.0/rate, self.timer_callback)
        
        self.get_logger().info('Node initialized')
    
    def callback(self, msg):
        """Handle incoming messages"""
        self.get_logger().info(f'Received: {msg.data}')
    
    def timer_callback(self):
        """Periodic task"""
        msg = String()
        msg.data = 'Hello'
        self.publisher.publish(msg)
```

---

## Complete Example Project {#example-project}

Here's a complete, production-ready package structure:

```
my_robot_controller/
├── my_robot_controller/
│   ├── __init__.py
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── velocity_controller.py
│   │   ├── position_tracker.py
│   │   └── safety_monitor.py
│   └── utils/
│       ├── __init__.py
│       ├── math_utils.py
│       └── ros_utils.py
├── launch/
│   ├── robot.launch.py
│   ├── simulation.launch.py
│   └── real_robot.launch.py
├── config/
│   ├── params.yaml
│   ├── sim_params.yaml
│   └── robot.rviz
├── test/
│   ├── test_velocity_controller.py
│   ├── test_position_tracker.py
│   └── test_utils.py
├── docs/
│   ├── README.md
│   └── API.md
├── package.xml
├── setup.py
├── setup.cfg
└── README.md
```

This structure is clean, maintainable, and follows ROS 2 best practices.

---

## Summary

Building well-structured ROS 2 packages is essential for creating maintainable, scalable robot systems. Proper organization, clear dependencies, and comprehensive testing set the foundation for successful robotics projects.

**Key Takeaways:**

1. ✅ Choose appropriate build system (ament_python vs ament_cmake)
2. ✅ Organize code into logical modules
3. ✅ Use launch files for complex startup
4. ✅ Manage configuration with YAML files
5. ✅ Write tests for your nodes
6. ✅ Document your packages

**Next Chapter:**

Chapter 6 will introduce Gazebo simulation, where we'll learn to test our packages in realistic virtual environments before deploying to real hardware.

---

*This chapter taught you how to create, organize, and build professional ROS 2 packages. You're now ready to develop complex robot applications with proper software engineering practices.*