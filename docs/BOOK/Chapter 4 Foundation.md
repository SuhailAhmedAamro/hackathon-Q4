# Chapter 4: Topics, Services, and Actions in ROS 2

> "Communication is the lifeblood of robotics. Nodes must share information seamlessly to create intelligent, coordinated behavior."

## Table of Contents

1. [Introduction to ROS 2 Communication](#introduction)
2. [Deep Dive: Topics](#topics)
3. [Deep Dive: Services](#services)
4. [Deep Dive: Actions](#actions)
5. [Custom Messages and Interfaces](#custom-messages)
6. [Communication Patterns](#patterns)
7. [Performance Considerations](#performance)
8. [Debugging Communication](#debugging)
9. [Real-World Examples](#examples)

---

## Introduction to ROS 2 Communication {#introduction}

ROS 2 provides three primary communication mechanisms, each designed for specific use cases. Understanding when and how to use each pattern is crucial for building efficient robot systems.

### Communication Paradigms Comparison

| Aspect | Topics | Services | Actions |
|--------|--------|----------|---------|
| **Pattern** | Publish-Subscribe | Request-Reply | Goal-Feedback-Result |
| **Direction** | One-to-many | One-to-one | One-to-one |
| **Synchronization** | Asynchronous | Synchronous | Asynchronous |
| **Duration** | Continuous | Instant | Long-running |
| **Feedback** | No | No | Yes |
| **Cancellation** | N/A | No | Yes |
| **Buffering** | Yes (QoS) | No | Yes |

### Choosing the Right Mechanism

**Use Topics for:**
- 🎥 Sensor data streams (camera, LiDAR, IMU)
- 🤖 Robot state (position, velocity, joint states)
- 📡 Telemetry and monitoring
- 🔄 Continuous data flow

**Use Services for:**
- ⚙️ Configuration changes
- 🔄 State queries
- 🎮 One-time commands
- 💾 Data requests

**Use Actions for:**
- 🎯 Navigation to goal
- 🤏 Object manipulation
- 📹 Trajectory execution
- ⏱️ Any task requiring feedback

---

## Deep Dive: Topics {#topics}

Topics implement a publish-subscribe pattern where publishers produce data and subscribers consume it, with no direct connection between them.

### Topic Architecture

```
┌─────────────┐      ┌─────────────┐
│ Publisher 1 │──┐   │             │
└─────────────┘  │   │             │
                 ├──→│   Topic     │──┐   ┌──────────────┐
┌─────────────┐  │   │   /scan     │  ├──→│ Subscriber 1 │
│ Publisher 2 │──┘   │             │  │   └──────────────┘
└─────────────┘      │             │  │   ┌──────────────┐
                     └─────────────┘  └──→│ Subscriber 2 │
                                          └──────────────┘
```

### Creating Publishers

**Basic Publisher:**

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class StringPublisher(Node):
    def __init__(self):
        super().__init__('string_publisher')
        
        # Create publisher
        self.publisher = self.create_publisher(
            String,           # Message type
            'chatter',        # Topic name
            10)              # QoS history depth
        
        # Timer for periodic publishing
        self.timer = self.create_timer(1.0, self.publish_message)
        self.counter = 0
    
    def publish_message(self):
        msg = String()
        msg.data = f'Hello ROS 2! Count: {self.counter}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Published: "{msg.data}"')
        self.counter += 1

def main():
    rclpy.init()
    node = StringPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

**Publisher with Custom QoS:**

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class SensorPublisher(Node):
    def __init__(self):
        super().__init__('sensor_publisher')
        
        # Custom QoS for sensor data
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        
        self.publisher = self.create_publisher(
            LaserScan,
            '/scan',
            qos_profile)
```

### Creating Subscribers

**Basic Subscriber:**

```python
from std_msgs.msg import String

class StringSubscriber(Node):
    def __init__(self):
        super().__init__('string_subscriber')
        
        # Create subscriber
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
    
    def listener_callback(self, msg):
        self.get_logger().info(f'Received: "{msg.data}"')
```

**Subscriber with Multiple Topics:**

```python
from sensor_msgs.msg import LaserScan, Image

class MultiSubscriber(Node):
    def __init__(self):
        super().__init__('multi_subscriber')
        
        # Subscribe to LiDAR
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10)
        
        # Subscribe to camera
        self.camera_sub = self.create_subscription(
            Image,
            '/camera/image',
            self.camera_callback,
            10)
        
        self.latest_scan = None
        self.latest_image = None
    
    def lidar_callback(self, msg):
        self.latest_scan = msg
        self.process_data()
    
    def camera_callback(self, msg):
        self.latest_image = msg
        self.process_data()
    
    def process_data(self):
        if self.latest_scan and self.latest_image:
            self.get_logger().info('Processing fused sensor data')
            # Perform sensor fusion here
```

### Message Synchronization

For applications requiring synchronized data from multiple topics:

```python
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image, CameraInfo

class SynchronizedSubscriber(Node):
    def __init__(self):
        super().__init__('synchronized_subscriber')
        
        # Create subscribers
        image_sub = Subscriber(self, Image, '/camera/image')
        info_sub = Subscriber(self, CameraInfo, '/camera/camera_info')
        
        # Synchronize with 0.1s tolerance
        self.sync = ApproximateTimeSynchronizer(
            [image_sub, info_sub],
            queue_size=10,
            slop=0.1)
        
        self.sync.registerCallback(self.synchronized_callback)
    
    def synchronized_callback(self, image_msg, info_msg):
        self.get_logger().info('Received synchronized messages')
        # Process synchronized data
```

### Topic Remapping

Remap topic names at runtime:

```bash
# Remap /cmd_vel to /robot1/cmd_vel
ros2 run my_package my_node --ros-args -r /cmd_vel:=/robot1/cmd_vel
```

### Topic Monitoring

```bash
# List all topics
ros2 topic list

# Show topic type
ros2 topic type /scan

# Display topic info
ros2 topic info /scan

# Echo messages
ros2 topic echo /scan

# Show publish rate
ros2 topic hz /scan

# Show bandwidth
ros2 topic bw /scan

# Publish from command line
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.5}, angular: {z: 0.2}}"
```

---

## Deep Dive: Services {#services}

Services implement request-reply communication, where a client sends a request and waits for a response from a server.

### Service Architecture

```
┌────────────┐   Request    ┌────────────┐
│   Client   │─────────────→│   Server   │
│            │              │            │
│            │←─────────────│            │
└────────────┘   Response   └────────────┘
```

### Creating Service Servers

**Basic Service Server:**

```python
from example_interfaces.srv import AddTwoInts

class AdditionServer(Node):
    def __init__(self):
        super().__init__('addition_server')
        
        # Create service
        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_callback)
        
        self.get_logger().info('Addition service ready')
    
    def add_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(
            f'Request: {request.a} + {request.b} = {response.sum}')
        return response
```

**Advanced Service Server with Validation:**

```python
from std_srvs.srv import SetBool

class MotorControlServer(Node):
    def __init__(self):
        super().__init__('motor_control')
        
        self.motors_enabled = False
        
        self.srv = self.create_service(
            SetBool,
            'enable_motors',
            self.enable_callback)
    
    def enable_callback(self, request, response):
        # Validate request
        if request.data == self.motors_enabled:
            response.success = True
            response.message = f'Motors already {"enabled" if request.data else "disabled"}'
            return response
        
        # Simulate motor control
        try:
            self.motors_enabled = request.data
            response.success = True
            response.message = f'Motors {"enabled" if request.data else "disabled"} successfully'
            self.get_logger().info(response.message)
        except Exception as e:
            response.success = False
            response.message = f'Failed to control motors: {str(e)}'
            self.get_logger().error(response.message)
        
        return response
```

### Creating Service Clients

**Synchronous Client:**

```python
from example_interfaces.srv import AddTwoInts

class AdditionClient(Node):
    def __init__(self):
        super().__init__('addition_client')
        
        # Create client
        self.client = self.create_client(AddTwoInts, 'add_two_ints')
        
        # Wait for service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for service...')
    
    def send_request(self, a, b):
        request = AddTwoInts.Request()
        request.a = a
        request.b = b
        
        # Call service (blocking)
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            response = future.result()
            self.get_logger().info(f'Result: {response.sum}')
            return response.sum
        else:
            self.get_logger().error('Service call failed')
            return None

def main():
    rclpy.init()
    client = AdditionClient()
    result = client.send_request(5, 7)
    client.destroy_node()
    rclpy.shutdown()
```

**Asynchronous Client with Callback:**

```python
class AsyncClient(Node):
    def __init__(self):
        super().__init__('async_client')
        self.client = self.create_client(AddTwoInts, 'add_two_ints')
    
    def send_request_async(self, a, b):
        request = AddTwoInts.Request()
        request.a = a
        request.b = b
        
        # Send request asynchronously
        future = self.client.call_async(request)
        future.add_done_callback(self.response_callback)
    
    def response_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f'Got result: {response.sum}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')
```

### Service Best Practices

**✅ DO:**
- Keep service calls fast (under 1 second)
- Return meaningful error messages
- Validate inputs
- Handle exceptions gracefully
- Use services for configuration, not streaming

**❌ DON'T:**
- Use services for high-frequency data
- Block for long periods
- Call services from callbacks
- Ignore timeouts
- Use services for real-time control

---

## Deep Dive: Actions {#actions}

Actions are designed for long-running tasks that require feedback and can be cancelled.

### Action Architecture

```
┌────────────┐   Goal       ┌────────────┐
│   Client   │─────────────→│   Server   │
│            │              │            │
│            │←─────────────│            │ (Feedback)
│            │   Feedback   │            │
│            │←─────────────│            │
│            │              │            │
│            │←─────────────│            │ (Result)
└────────────┘   Result     └────────────┘
        │
        └──→ Cancel (optional)
```

### Action Components

Every action has three parts:

1. **Goal**: The objective to achieve
2. **Feedback**: Progress updates during execution
3. **Result**: Final outcome

### Creating Action Servers

```python
from action_tutorials_interfaces.action import Fibonacci
from rclpy.action import ActionServer

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_server')
        
        # Create action server
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)
        
        self.get_logger().info('Fibonacci action server ready')
    
    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        
        # Initialize feedback
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]
        
        # Execute action
        for i in range(1, goal_handle.request.order):
            # Check if goal was cancelled
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal cancelled')
                return Fibonacci.Result()
            
            # Compute next Fibonacci number
            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])
            
            # Publish feedback
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Feedback: {feedback_msg.sequence}')
            
            time.sleep(0.5)  # Simulate work
        
        # Mark goal as succeeded
        goal_handle.succeed()
        
        # Return result
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        return result
```

### Action Server with Goal Policies

```python
from rclpy.action import ActionServer, GoalResponse, CancelResponse

class NavigationActionServer(Node):
    def __init__(self):
        super().__init__('navigation_server')
        
        self._action_server = ActionServer(
            self,
            NavigateToPose,
            'navigate_to_pose',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)
        
        self.current_goal = None
    
    def goal_callback(self, goal_request):
        """Accept or reject incoming goals"""
        self.get_logger().info('Received goal request')
        
        # Validate goal
        if self.current_goal is not None:
            self.get_logger().warn('Already executing a goal')
            return GoalResponse.REJECT
        
        # Check if goal position is valid
        if not self.is_valid_position(goal_request.pose):
            self.get_logger().warn('Invalid goal position')
            return GoalResponse.REJECT
        
        return GoalResponse.ACCEPT
    
    def cancel_callback(self, goal_handle):
        """Handle cancellation requests"""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT
    
    def execute_callback(self, goal_handle):
        self.current_goal = goal_handle
        
        # Execute navigation
        feedback_msg = NavigateToPose.Feedback()
        
        for step in range(100):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.current_goal = None
                return NavigateToPose.Result()
            
            # Update position
            feedback_msg.current_pose = self.get_current_pose()
            feedback_msg.distance_remaining = self.calculate_distance()
            goal_handle.publish_feedback(feedback_msg)
            
            time.sleep(0.1)
        
        goal_handle.succeed()
        self.current_goal = None
        
        result = NavigateToPose.Result()
        result.success = True
        return result
```

### Creating Action Clients

```python
from rclpy.action import ActionClient

class FibonacciActionClient(Node):
    def __init__(self):
        super().__init__('fibonacci_client')
        
        # Create action client
        self._action_client = ActionClient(
            self,
            Fibonacci,
            'fibonacci')
    
    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order
        
        # Wait for server
        self._action_client.wait_for_server()
        
        # Send goal
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)
        
        self._send_goal_future.add_done_callback(self.goal_response_callback)
    
    def goal_response_callback(self, future):
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return
        
        self.get_logger().info('Goal accepted')
        
        # Get result
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)
    
    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Feedback: {feedback.sequence}')
    
    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')

def main():
    rclpy.init()
    client = FibonacciActionClient()
    client.send_goal(10)
    rclpy.spin(client)
```

### Canceling Goals

```python
class CancellableClient(Node):
    def __init__(self):
        super().__init__('cancellable_client')
        self._action_client = ActionClient(self, Fibonacci, 'fibonacci')
        self._goal_handle = None
    
    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order
        
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)
    
    def goal_response_callback(self, future):
        self._goal_handle = future.result()
        
        if not self._goal_handle.accepted:
            return
        
        # Schedule cancellation after 2 seconds
        self.create_timer(2.0, self.cancel_goal)
    
    def cancel_goal(self):
        if self._goal_handle is not None:
            self.get_logger().info('Canceling goal')
            cancel_future = self._goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self.cancel_done)
    
    def cancel_done(self, future):
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) > 0:
            self.get_logger().info('Goal successfully canceled')
        else:
            self.get_logger().info('Goal failed to cancel')
```

---

## Custom Messages and Interfaces {#custom-messages}

Create custom message types for application-specific data.

### Message Definition

```bash
# Create package for interfaces
ros2 pkg create my_robot_interfaces --build-type ament_cmake
```

**Custom Message (msg/RobotStatus.msg):**

```
# RobotStatus.msg
string robot_name
float32 battery_level
float32 temperature
bool motors_enabled
geometry_msgs/Pose current_pose
```

**Custom Service (srv/SetMode.srv):**

```
# SetMode.srv
string mode
---
bool success
string message
```

**Custom Action (action/Navigate.action):**

```
# Navigate.action
# Goal
geometry_msgs/PoseStamped target_pose
---
# Result
bool success
float32 total_distance
---
# Feedback
geometry_msgs/PoseStamped current_pose
float32 distance_remaining
float32 estimated_time
```

### CMakeLists.txt Configuration

```cmake
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RobotStatus.msg"
  "srv/SetMode.srv"
  "action/Navigate.action"
  DEPENDENCIES geometry_msgs
)
```

### Package.xml Dependencies

```xml
<buildtool_depend>rosidl_default_generators</buildtool_depend>
<exec_depend>rosidl_default_runtime</exec_depend>
<member_of_group>rosidl_interface_packages</member_of_group>

<depend>geometry_msgs</depend>
```

### Using Custom Interfaces

```python
from my_robot_interfaces.msg import RobotStatus
from my_robot_interfaces.srv import SetMode
from my_robot_interfaces.action import Navigate

class RobotStatusPublisher(Node):
    def __init__(self):
        super().__init__('status_publisher')
        
        self.publisher = self.create_publisher(
            RobotStatus,
            'robot_status',
            10)
        
        self.timer = self.create_timer(1.0, self.publish_status)
    
    def publish_status(self):
        msg = RobotStatus()
        msg.robot_name = 'Robot_01'
        msg.battery_level = 85.5
        msg.temperature = 45.2
        msg.motors_enabled = True
        
        self.publisher.publish(msg)
```

---

## Communication Patterns {#patterns}

### Pattern 1: Pipeline Processing

```
[Camera] → /image → [Detector] → /detections → [Tracker] → /tracked_objects
```

```python
class PipelineNode(Node):
    def __init__(self):
        super().__init__('pipeline_node')
        
        # Input
        self.create_subscription(Image, '/image', self.process, 10)
        
        # Output
        self.pub = self.create_publisher(DetectionArray, '/detections', 10)
    
    def process(self, image_msg):
        # Process image
        detections = self.detect_objects(image_msg)
        
        # Publish results
        self.pub.publish(detections)
```

### Pattern 2: Request-Reply

```
[Client] ──Request──→ [Server]
[Client] ←─Response── [Server]
```

### Pattern 3: Goal-Oriented

```
[Client] ──Goal────→ [Server]
[Client] ←─Feedback─ [Server] (periodic)
[Client] ←─Result──  [Server] (final)
```

### Pattern 4: Sensor Fusion

```
[Camera]  → /image ──┐
                     ├─→ [Fusion] → /fused_data
[LiDAR]   → /scan ───┘
```

---

## Performance Considerations {#performance}

### Latency Optimization

**Minimize callback execution time:**

```python
class FastSubscriber(Node):
    def __init__(self):
        super().__init__('fast_subscriber')
        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.callback, 10)
        self.processing_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self.process_loop)
        self.processing_thread.start()
    
    def callback(self, msg):
        # Quick callback - just queue the message
        self.processing_queue.put(msg)
    
    def process_loop(self):
        # Heavy processing in separate thread
        while rclpy.ok():
            msg = self.processing_queue.get()
            self.heavy_processing(msg)
```

### Bandwidth Optimization

**Use appropriate QoS:**

```python
# For sensor data (best-effort, low latency)
sensor_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1)

# For critical data (reliable, no loss)
critical_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_ALL)
```

### CPU Usage Optimization

**Adjust publish rates:**

```python
# High-frequency sensor (100 Hz)
self.create_timer(0.01, self.publish_fast)

# Low-frequency status (1 Hz)
self.create_timer(1.0, self.publish_slow)
```

---

## Debugging Communication {#debugging}

### Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| **No data received** | Subscriber silent | Check topic names, QoS compatibility |
| **Delayed messages** | Old timestamps | Adjust QoS depth, check network |
| **High CPU usage** | System slow | Reduce publish rate, optimize callbacks |
| **Memory leak** | RAM increases | Check for unreleased resources |

### Debugging Tools

```bash
# Monitor topic bandwidth
ros2 topic bw /scan

# Check message latency
ros2 topic delay /scan

# Visualize computation graph
rqt_graph

# Record and analyze
ros2 bag record -a
ros2 bag play --rate 0.5 mybag.db3
```

---

## Real-World Examples {#examples}

### Example 1: Robot Teleoperation

```python
class Teleop(Node):
    def __init__(self):
        super().__init__('teleop')
        
        # Publish velocity commands
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscribe to joystick
        self.joy_sub = self.create_subscription(
            Joy, '/joy', self.joy_callback, 10)
    
    def joy_callback(self, joy_msg):
        cmd = Twist()
        cmd.linear.x = joy_msg.axes[1] * 0.5  # Forward/backward
        cmd.angular.z = joy_msg.axes[0] * 1.0  # Left/right
        self.cmd_pub.publish(cmd)
```

### Example 2: Safety Monitor

```python
class SafetyMonitor(Node):
    def __init__(self):
        super().__init__('safety_monitor')
        
        # Subscribe to sensors
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        
        # Emergency stop service
        self.estop_client = self.create_client(SetBool, '/emergency_stop')
    
    def scan_callback(self, scan_msg):
        min_distance = min(scan_msg.ranges)
        
        if min_distance < 0.3:  # 30cm threshold
            self.get_logger().warn('COLLISION WARNING!')
            self.trigger_estop()
    
    def trigger_estop(self):
        request = SetBool.Request()
        request.data = True
        self.estop_client.call_async(request)
```

---

## Summary

ROS 2's communication mechanisms provide flexible, powerful ways to build distributed robot systems. Understanding topics, services, and actions—and when to use each—is fundamental to effective robotics software development.

**Key Takeaways:**

1. ✅ Topics for continuous data streams
2. ✅ Services for request-reply interactions
3. ✅ Actions for long-running goal-oriented tasks
4. ✅ QoS policies control message delivery
5. ✅ Custom interfaces for application-specific data
6. ✅ Performance requires careful design choices

**Next Chapter:**

Chapter 5 will guide you through building complete ROS 2 packages with proper structure, launch files, and configuration management.

---

*This chapter explored ROS 2 communication in depth. You now understand how to effectively use topics, services, and actions to build sophisticated robot behaviors.*