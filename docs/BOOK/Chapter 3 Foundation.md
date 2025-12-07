# Chapter 3: ROS 2 Architecture

> "ROS is the nervous system of modern robotics—connecting sensors, algorithms, and actuators into a unified intelligent system."

## Table of Contents

1. [Introduction to ROS 2](#introduction)
2. [Core Concepts](#core-concepts)
3. [Communication Patterns](#communication)
4. [Computation Graph](#computation-graph)
5. [Quality of Service (QoS)](#qos)
6. [ROS 2 vs ROS 1](#comparison)
7. [Installation and Setup](#installation)
8. [Your First ROS 2 Node](#first-node)
9. [Command Line Tools](#cli-tools)
10. [Best Practices](#best-practices)

---

## Introduction to ROS 2 {#introduction}

**ROS (Robot Operating System)** is not actually an operating system—it's a flexible framework and middleware for writing robot software. ROS 2 is the next generation, redesigned from the ground up to address the limitations of ROS 1.

### What is ROS 2?

ROS 2 provides:

- **Communication infrastructure** for distributed robot systems
- **Standard message formats** for sensors and actuators  
- **Tools** for visualization, debugging, and simulation
- **Libraries** for common robotics algorithms
- **Hardware abstraction** layer

### Why ROS 2?

**Improvements over ROS 1:**

| Feature | ROS 1 | ROS 2 |
|---------|-------|-------|
| **Real-time** | No guarantees | Real-time capable |
| **Security** | None | DDS security |
| **Multi-robot** | Difficult | Native support |
| **Embedded** | Limited | Microcontroller support |
| **Communication** | Custom TCPROS | Industry-standard DDS |
| **Lifecycle** | Basic | Advanced node lifecycle |
| **QoS** | Limited | Configurable QoS policies |

### When to Use ROS 2

**Perfect for:**
- ✅ Research and prototyping
- ✅ Multi-robot systems
- ✅ Real-time control
- ✅ Production robotics
- ✅ Safety-critical applications

**Not ideal for:**
- ❌ Simple single-purpose devices
- ❌ Extremely resource-constrained systems
- ❌ Non-robotics applications

---

## Core Concepts {#core-concepts}

ROS 2 is built around several fundamental concepts that form the basis of all robot applications.

### Nodes

A **node** is a single executable program that performs computation. Nodes are the building blocks of ROS systems.

**Characteristics:**
- Self-contained process
- Performs a specific task
- Communicates with other nodes
- Can be written in Python, C++, or other languages

**Example Node Types:**

| Node Type | Purpose | Example |
|-----------|---------|---------|
| **Driver** | Hardware interface | Camera driver, motor controller |
| **Algorithm** | Data processing | Object detection, SLAM |
| **Controller** | Robot control | Navigation, manipulation |
| **Tool** | Utilities | Logger, monitor, visualizer |

```python
# Minimal ROS 2 node in Python
import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_node')
        self.get_logger().info('Node started!')

def main():
    rclpy.init()
    node = MinimalNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Topics

**Topics** enable continuous data streaming between nodes using a publish-subscribe pattern.

**How Topics Work:**

```
Publisher Node → Topic → Subscriber Node(s)
                  ↓
            Message Type
```

**Characteristics:**
- **Many-to-many**: Multiple publishers and subscribers
- **Anonymous**: Publishers don't know about subscribers
- **Asynchronous**: Non-blocking communication
- **Typed**: Each topic has a message type

**Common Topic Examples:**

| Topic Name | Message Type | Purpose |
|------------|--------------|---------|
| `/camera/image` | `sensor_msgs/Image` | Camera feed |
| `/cmd_vel` | `geometry_msgs/Twist` | Robot velocity |
| `/scan` | `sensor_msgs/LaserScan` | LiDAR data |
| `/odom` | `nav_msgs/Odometry` | Robot odometry |
| `/joint_states` | `sensor_msgs/JointState` | Joint positions |

```python
# Publisher example
from geometry_msgs.msg import Twist

class VelocityPublisher(Node):
    def __init__(self):
        super().__init__('velocity_publisher')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.publish_velocity)
    
    def publish_velocity(self):
        msg = Twist()
        msg.linear.x = 0.5  # 0.5 m/s forward
        msg.angular.z = 0.0  # No rotation
        self.publisher.publish(msg)
```

```python
# Subscriber example
class VelocitySubscriber(Node):
    def __init__(self):
        super().__init__('velocity_subscriber')
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.velocity_callback,
            10)
    
    def velocity_callback(self, msg):
        self.get_logger().info(
            f'Linear: {msg.linear.x:.2f}, Angular: {msg.angular.z:.2f}')
```

### Services

**Services** implement synchronous request-response communication.

**How Services Work:**

```
Client Node → Request → Service Server
                          ↓
Client Node ← Response ← Service Server
```

**Characteristics:**
- **One-to-one**: Single client, single server
- **Synchronous**: Client waits for response
- **Typed**: Request and response types
- **Ephemeral**: No data persistence

**Common Service Examples:**

| Service Name | Type | Purpose |
|--------------|------|---------|
| `/reset_simulation` | `std_srvs/Empty` | Reset simulator |
| `/spawn_entity` | `gazebo_msgs/SpawnEntity` | Spawn robot |
| `/set_parameters` | `rcl_interfaces/SetParameters` | Configure node |
| `/get_map` | `nav_msgs/GetMap` | Request map data |

```python
# Service server example
from std_srvs.srv import SetBool

class ServiceServer(Node):
    def __init__(self):
        super().__init__('service_server')
        self.srv = self.create_service(
            SetBool,
            '/enable_motors',
            self.enable_callback)
    
    def enable_callback(self, request, response):
        if request.data:
            self.get_logger().info('Motors enabled')
            response.success = True
            response.message = 'Motors are now ON'
        else:
            self.get_logger().info('Motors disabled')
            response.success = True
            response.message = 'Motors are now OFF'
        return response
```

```python
# Service client example
class ServiceClient(Node):
    def __init__(self):
        super().__init__('service_client')
        self.client = self.create_client(SetBool, '/enable_motors')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for service...')
    
    def send_request(self, enable):
        request = SetBool.Request()
        request.data = enable
        future = self.client.call_async(request)
        return future
```

### Actions

**Actions** are for long-running tasks that need feedback and can be preempted.

**How Actions Work:**

```
Action Client → Goal → Action Server
               ↓
Action Client ← Feedback ← Action Server (periodic)
               ↓
Action Client ← Result ← Action Server (final)
```

**Characteristics:**
- **Asynchronous**: Non-blocking
- **Feedback**: Progress updates
- **Cancellable**: Can be preempted
- **Goal-oriented**: Specific objectives

**Example Actions:**

| Action Name | Type | Purpose |
|-------------|------|---------|
| `/navigate_to_pose` | `NavigateToPose` | Move to goal |
| `/follow_path` | `FollowPath` | Follow trajectory |
| `/pick_object` | `PickPlace` | Grasp object |

```python
# Action server example
from action_msgs.msg import GoalStatus
from example_interfaces.action import Fibonacci

class FibonacciServer(Node):
    def __init__(self):
        super().__init__('fibonacci_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)
    
    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        
        # Initialize feedback
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]
        
        # Compute Fibonacci sequence
        for i in range(1, goal_handle.request.order):
            # Check if canceled
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return Fibonacci.Result()
            
            # Calculate next number
            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])
            
            # Publish feedback
            goal_handle.publish_feedback(feedback_msg)
            time.sleep(0.1)
        
        # Set result
        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        return result
```

### Parameters

**Parameters** are configuration values that nodes can get and set at runtime.

**Parameter Types:**
- `bool`
- `integer`
- `double`
- `string`
- `byte_array`
- `bool_array`, `integer_array`, `double_array`, `string_array`

```python
# Using parameters
class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')
        
        # Declare parameters with defaults
        self.declare_parameter('robot_name', 'my_robot')
        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('use_lidar', True)
        
        # Get parameter values
        robot_name = self.get_parameter('robot_name').value
        max_speed = self.get_parameter('max_speed').value
        use_lidar = self.get_parameter('use_lidar').value
        
        self.get_logger().info(f'Robot: {robot_name}')
        self.get_logger().info(f'Max speed: {max_speed} m/s')
        self.get_logger().info(f'LiDAR enabled: {use_lidar}')
```

**Setting parameters from command line:**

```bash
ros2 run my_package my_node --ros-args -p robot_name:=robot1 -p max_speed:=2.0
```

---

## Communication Patterns {#communication}

ROS 2 supports different communication patterns for different use cases.

### Pattern Comparison

| Pattern | Use Case | Latency | Guarantee |
|---------|----------|---------|-----------|
| **Topic** | Streaming data | Low | Best-effort or reliable |
| **Service** | Request-response | Medium | Reliable |
| **Action** | Long tasks | Medium-High | Reliable with feedback |

### When to Use Each

**Use Topics when:**
- Continuous data stream (sensor readings)
- Multiple subscribers needed
- Data can be dropped occasionally
- Low latency required

**Use Services when:**
- One-time requests
- Response required
- Synchronous operation acceptable
- Configuration or control

**Use Actions when:**
- Long-running operations
- Progress feedback needed
- Cancellation support required
- Goal-oriented tasks

---

## Computation Graph {#computation-graph}

The **computation graph** is the network of nodes and their communication links.

### Graph Components

```
        Topic: /scan
           ↓
[LiDAR Node] → Topic: /cloud → [SLAM Node] → Topic: /map
                                     ↓
                              Service: /save_map
```

### Visualizing the Graph

```bash
# Install rqt_graph
sudo apt install ros-humble-rqt-graph

# Run visualization
rqt_graph
```

### Graph Introspection

```bash
# List all nodes
ros2 node list

# List all topics
ros2 topic list

# Show node info
ros2 node info /my_node

# Show topic info
ros2 topic info /scan

# Show topic type
ros2 topic type /scan
```

---

## Quality of Service (QoS) {#qos}

QoS policies control how messages are delivered in ROS 2.

### QoS Policies

| Policy | Options | Purpose |
|--------|---------|---------|
| **Reliability** | Best-effort, Reliable | Delivery guarantee |
| **Durability** | Volatile, Transient-local | Message persistence |
| **History** | Keep-last, Keep-all | Message buffering |
| **Depth** | Number (e.g., 10) | Buffer size |
| **Deadline** | Duration | Message frequency |
| **Lifespan** | Duration | Message validity |
| **Liveliness** | Automatic, Manual | Publisher aliveness |

### QoS Profiles

**Predefined profiles:**

```python
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# Sensor data (best-effort, volatile)
sensor_qos = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10
)

# Service-like (reliable, transient-local)
service_qos = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10
)

# Create publisher with custom QoS
publisher = self.create_publisher(LaserScan, '/scan', sensor_qos)
```

### Common QoS Profiles

| Profile | Use Case |
|---------|----------|
| **Sensor Data** | High-frequency sensor readings |
| **Parameters** | Configuration values |
| **Services** | Request-response communication |
| **System Default** | General purpose |

---

## ROS 2 vs ROS 1 {#comparison}

### Architecture Differences

| Aspect | ROS 1 | ROS 2 |
|--------|-------|-------|
| **Master** | Required (roscore) | Not needed |
| **Discovery** | Centralized | Distributed (DDS) |
| **Python** | Python 2.7 | Python 3.6+ |
| **Build System** | catkin | ament/colcon |
| **Message Format** | .msg files | .msg/.idl files |
| **Middleware** | Custom TCPROS | DDS (pluggable) |

### Migration Considerations

**Code Changes:**
- `rospy` → `rclpy` (Python) or `rclcpp` (C++)
- Package format: `package.xml` format 3
- Launch files: Python-based instead of XML
- Build system: `colcon` instead of `catkin`

**Compatibility:**
- ROS 1 Bridge available for gradual migration
- No direct runtime compatibility
- Message conversion needed

---

## Installation and Setup {#installation}

### Installing ROS 2 Humble (Ubuntu 22.04)

```bash
# Set up sources
sudo apt update && sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) \
    signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu \
    $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2
sudo apt update
sudo apt upgrade
sudo apt install ros-humble-desktop

# Install development tools
sudo apt install ros-dev-tools

# Source ROS 2
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Workspace Setup

```bash
# Create workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Build workspace
colcon build

# Source workspace
source install/setup.bash
```

---

## Your First ROS 2 Node {#first-node}

Let's create a complete ROS 2 package with a publisher and subscriber.

### Create Package

```bash
cd ~/ros2_ws/src
ros2 pkg create my_robot_controller \
    --build-type ament_python \
    --dependencies rclpy std_msgs geometry_msgs
```

### Publisher Node

```python
# my_robot_controller/my_robot_controller/publisher_node.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class VelocityPublisher(Node):
    def __init__(self):
        super().__init__('velocity_publisher')
        
        # Create publisher
        self.publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10)
        
        # Create timer (10 Hz)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.counter = 0
        
        self.get_logger().info('Velocity Publisher started')
    
    def timer_callback(self):
        msg = Twist()
        msg.linear.x = 0.5
        msg.angular.z = 0.2 * (1 if self.counter % 2 == 0 else -1)
        
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: linear={msg.linear.x}, '
                              f'angular={msg.angular.z}')
        self.counter += 1

def main(args=None):
    rclpy.init(args=args)
    node = VelocityPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Subscriber Node

```python
# my_robot_controller/my_robot_controller/subscriber_node.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class VelocitySubscriber(Node):
    def __init__(self):
        super().__init__('velocity_subscriber')
        
        # Create subscriber
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.velocity_callback,
            10)
        
        self.get_logger().info('Velocity Subscriber started')
    
    def velocity_callback(self, msg):
        self.get_logger().info(
            f'Received: linear={msg.linear.x:.2f}, '
            f'angular={msg.angular.z:.2f}')

def main(args=None):
    rclpy.init(args=args)
    node = VelocitySubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Setup.py Configuration

```python
# my_robot_controller/setup.py
from setuptools import setup

package_name = 'my_robot_controller'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='My first ROS 2 package',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'publisher = my_robot_controller.publisher_node:main',
            'subscriber = my_robot_controller.subscriber_node:main',
        ],
    },
)
```

### Build and Run

```bash
# Build package
cd ~/ros2_ws
colcon build --packages-select my_robot_controller
source install/setup.bash

# Run publisher (Terminal 1)
ros2 run my_robot_controller publisher

# Run subscriber (Terminal 2)
ros2 run my_robot_controller subscriber
```

---

## Command Line Tools {#cli-tools}

### Essential Commands

**Node commands:**
```bash
ros2 node list                    # List running nodes
ros2 node info /node_name         # Node details
```

**Topic commands:**
```bash
ros2 topic list                   # List all topics
ros2 topic echo /topic_name       # Print messages
ros2 topic hz /topic_name         # Message frequency
ros2 topic pub /topic_name ...    # Publish message
```

**Service commands:**
```bash
ros2 service list                 # List services
ros2 service call /service_name   # Call service
```

**Parameter commands:**
```bash
ros2 param list                   # List parameters
ros2 param get /node_name param   # Get parameter
ros2 param set /node_name param value  # Set parameter
```

**Bag commands (recording):**
```bash
ros2 bag record -a                # Record all topics
ros2 bag play bag_file            # Replay recording
ros2 bag info bag_file            # Show recording info
```

---

## Best Practices {#best-practices}

### Node Design

**✅ DO:**
- Keep nodes small and focused
- Use meaningful names
- Log important events
- Handle errors gracefully
- Implement clean shutdown

**❌ DON'T:**
- Create monolithic nodes
- Use hardcoded values (use parameters)
- Ignore error conditions
- Block in callbacks
- Use global variables excessively

### Communication

**Topic best practices:**
- Use standard message types when possible
- Choose appropriate QoS profiles
- Avoid high-frequency topics when not needed
- Namespace topics logically

**Service best practices:**
- Keep services fast (< 1 second)
- Return meaningful status codes
- Use services for configuration, not data streaming

**Action best practices:**
- Provide regular feedback
- Support cancellation
- Set reasonable timeouts

### Code Organization

```
my_robot_package/
├── my_robot_package/
│   ├── __init__.py
│   ├── nodes/           # Node implementations
│   ├── utils/           # Helper functions
│   └── config/          # Configuration files
├── launch/              # Launch files
├── config/              # YAML configs
├── package.xml
└── setup.py
```

---

## Summary

ROS 2 provides a robust, flexible framework for building modern robot systems. Its distributed architecture, real-time capabilities, and industry-standard middleware make it the platform of choice for both research and production robotics.

**Key Takeaways:**

1. ✅ ROS 2 is middleware, not an OS
2. ✅ Nodes communicate via topics, services, and actions
3. ✅ QoS policies control message delivery
4. ✅ No master node required (distributed discovery)
5. ✅ Python and C++ are primary languages
6. ✅ Excellent tools for debugging and visualization

**Next Chapter:**

In Chapter 4, we'll dive deeper into ROS 2 communication patterns and build more complex multi-node systems with topics, services, and actions.

---

*This chapter introduced ROS 2 architecture and core concepts. You've learned how to create packages, write nodes, and use the communication infrastructure that powers modern robotics.*