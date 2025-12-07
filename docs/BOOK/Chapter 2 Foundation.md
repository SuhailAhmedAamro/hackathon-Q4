# Chapter 2: Sensors & Perception in Physical AI

> "Without perception, there is no intelligence. A robot without sensors is like a mind without senses—capable of thought but unable to interact with reality."

## Table of Contents

1. [Introduction to Robot Perception](#introduction)
2. [Visual Sensors](#visual-sensors)
3. [Range and Depth Sensors](#range-sensors)
4. [Motion and Orientation Sensors](#motion-sensors)
5. [Force and Tactile Sensors](#force-sensors)
6. [Sensor Fusion](#sensor-fusion)
7. [Perception Pipeline](#perception-pipeline)
8. [Challenges in Real-World Perception](#challenges)
9. [Practical Applications](#applications)

---

## Introduction to Robot Perception {#introduction}

Robot perception is the process by which robots sense and interpret their environment. Just as humans rely on their five senses to understand the world, robots use various sensors to gather information about their surroundings. This sensory data forms the foundation for all intelligent behavior—from simple obstacle avoidance to complex manipulation tasks.

### Why Perception Matters

Perception enables robots to:

- **Navigate** safely through dynamic environments
- **Identify** objects, people, and obstacles
- **Manipulate** objects with appropriate force and precision
- **Interact** naturally with humans
- **Adapt** to changing conditions in real-time

### The Perception Challenge

Unlike controlled factory environments, real-world settings present numerous challenges:

- 🌦️ Varying lighting conditions (bright sun to darkness)
- 🏃 Dynamic obstacles (moving people, vehicles)
- 🎭 Object variations (different shapes, colors, textures)
- 📏 Scale differences (small screws to large furniture)
- ⚡ Real-time requirements (decisions in milliseconds)

---

## Visual Sensors {#visual-sensors}

Vision is arguably the most information-rich sensing modality for robots, providing detailed information about the environment's appearance, structure, and composition.

### RGB Cameras

Standard color cameras capture images similar to human vision.

**Specifications:**
- **Resolution**: 640x480 (VGA) to 4K (3840x2160)
- **Frame Rate**: 30-120 fps
- **Field of View**: 60°-180°
- **Output**: RGB color images

**Advantages:**
- ✅ Rich color and texture information
- ✅ High resolution for detailed recognition
- ✅ Mature computer vision algorithms
- ✅ Relatively low cost

**Limitations:**
- ❌ No direct depth information
- ❌ Sensitive to lighting conditions
- ❌ Computationally intensive processing

**Applications:**
- Object recognition and classification
- Lane detection in autonomous vehicles
- Human pose estimation
- Visual servoing for manipulation

```python
# Example: Capturing images with OpenCV
import cv2

# Initialize camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    ret, frame = camera.read()
    if not ret:
        break
    
    # Process frame
    cv2.imshow('Robot Vision', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
```

### Depth Cameras

Depth cameras provide distance information for each pixel, creating a 3D representation of the scene.

#### Stereo Cameras

Use two cameras (like human eyes) to estimate depth through triangulation.

**How They Work:**
1. Capture two images from slightly different positions
2. Find corresponding points in both images
3. Calculate depth using disparity (difference in position)

**Popular Models:**
- Intel RealSense D400 series
- ZED 2 by Stereolabs
- OAK-D by Luxonis

**Example Configuration:**

| Parameter | Typical Value |
|-----------|---------------|
| Baseline | 50-100mm |
| Depth Range | 0.3-10m |
| Depth Accuracy | ±2% at 2m |
| Resolution | 1280x720 |

#### Structured Light Cameras

Project a known pattern (dots or lines) onto the scene and analyze deformation to compute depth.

**Advantages:**
- ✅ Works in low light
- ✅ High accuracy at close range
- ✅ Dense depth maps

**Limitations:**
- ❌ Limited range (typically under 5m)
- ❌ Struggles outdoors (sunlight interference)
- ❌ Multiple devices can interfere

**Example: Microsoft Kinect (now discontinued but influential)**

#### Time-of-Flight (ToF) Cameras

Measure the time light takes to travel to an object and back.

**Advantages:**
- ✅ Fast depth acquisition
- ✅ Simple depth calculation
- ✅ Works with moving objects

**Applications:**
- Gesture recognition
- 3D scanning
- Collision avoidance

```python
# Example: Using Intel RealSense for depth perception
import pyrealsense2 as rs
import numpy as np

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        # Convert to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Get distance at center point
        center_x, center_y = 320, 240
        distance = depth_frame.get_distance(center_x, center_y)
        print(f"Distance to center: {distance:.2f}m")

finally:
    pipeline.stop()
```

### Thermal Cameras

Detect infrared radiation (heat) emitted by objects.

**Use Cases:**
- Night vision for autonomous vehicles
- Human detection in low visibility
- Temperature monitoring in industrial settings
- Search and rescue operations

**Specifications:**
- **Resolution**: 320x240 to 640x512
- **Temperature Range**: -20°C to 500°C
- **Thermal Sensitivity**: ±2°C

---

## Range and Depth Sensors {#range-sensors}

### LiDAR (Light Detection and Ranging)

LiDAR uses laser pulses to measure distances, creating highly accurate 3D point clouds of the environment.

**How LiDAR Works:**

1. Emit laser pulse
2. Pulse reflects off object
3. Sensor detects return signal
4. Calculate distance: `distance = (speed_of_light × time) / 2`
5. Rotate to scan 360° (or use solid-state design)

**Types of LiDAR:**

| Type | Scan Pattern | Use Case |
|------|--------------|----------|
| **Mechanical** | 360° rotation | Autonomous vehicles, mapping |
| **Solid-State** | Fixed FOV | Compact applications |
| **MEMS** | Mirror-based | Cost-effective solution |
| **Flash** | Entire scene at once | Fast moving robots |

**Specifications Comparison:**

| Parameter | 2D LiDAR | 3D LiDAR |
|-----------|----------|----------|
| **Range** | 0.1-30m | 0.5-200m |
| **Angular Resolution** | 0.25°-1° | 0.1°-0.4° |
| **Scan Rate** | 5-15 Hz | 10-20 Hz |
| **Points per Second** | 10K-50K | 300K-2M |
| **Cost** | $100-1K | $1K-75K |

**Popular Models:**
- **Velodyne VLP-16**: 16 channels, 100m range, autonomous vehicles
- **Ouster OS1**: 64/128 channels, high resolution
- **Livox Mid-70**: Non-repetitive scanning, cost-effective
- **SICK TiM**: 2D scanning, industrial robotics

**Advantages:**
- ✅ Highly accurate (±2cm)
- ✅ Long range (up to 200m)
- ✅ Works in all lighting conditions
- ✅ Direct 3D information

**Limitations:**
- ❌ High cost (especially for dense 3D)
- ❌ Struggles with reflective/transparent surfaces
- ❌ Limited color information
- ❌ Moving parts (mechanical models)

```python
# Example: Processing LiDAR data in ROS 2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

class LidarProcessor(Node):
    def __init__(self):
        super().__init__('lidar_processor')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
    
    def scan_callback(self, msg):
        # Find minimum distance (closest obstacle)
        min_distance = min(msg.ranges)
        min_index = msg.ranges.index(min_distance)
        angle = msg.angle_min + min_index * msg.angle_increment
        
        self.get_logger().info(
            f'Closest obstacle: {min_distance:.2f}m at {angle:.2f} rad')
        
        # Check if obstacle is too close
        if min_distance < 0.5:
            self.get_logger().warn('DANGER: Obstacle too close!')

def main():
    rclpy.init()
    node = LidarProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

### Radar

Radio Detection and Ranging uses radio waves to detect objects and measure velocity.

**Advantages over LiDAR:**
- ✅ Works in fog, rain, dust
- ✅ Measures velocity directly (Doppler effect)
- ✅ Long range
- ✅ Lower cost

**Limitations:**
- ❌ Lower resolution than LiDAR
- ❌ Less accurate for static objects
- ❌ Reflections from metal surfaces

**Applications:**
- Automotive adaptive cruise control
- Collision avoidance
- Speed measurement
- Long-range detection

### Ultrasonic Sensors

Use sound waves (40 kHz) to measure distance.

**Characteristics:**
- **Range**: 2cm to 4m
- **Beam Width**: 15°-30° (cone-shaped)
- **Update Rate**: 10-20 Hz
- **Cost**: Very low ($2-20)

**Advantages:**
- ✅ Very inexpensive
- ✅ Simple to use
- ✅ Reliable for close-range detection

**Limitations:**
- ❌ Short range
- ❌ Affected by soft/angled surfaces
- ❌ Slow update rate
- ❌ Wide beam (poor directionality)

**Common Uses:**
- Parking sensors
- Obstacle detection for mobile robots
- Liquid level measurement
- Proximity detection

---

## Motion and Orientation Sensors {#motion-sensors}

### Inertial Measurement Unit (IMU)

IMUs measure acceleration and rotation, essential for understanding robot motion and orientation.

**Components:**

1. **Accelerometer** (3-axis)
   - Measures linear acceleration
   - Detects gravity direction
   - Senses vibration and shock

2. **Gyroscope** (3-axis)
   - Measures angular velocity
   - Detects rotation rate
   - High-frequency motion tracking

3. **Magnetometer** (3-axis, optional)
   - Measures magnetic field
   - Provides absolute heading
   - Compass functionality

**9-DOF IMU (Degrees of Freedom):**

| Sensor | Axes | Measurement |
|--------|------|-------------|
| Accelerometer | X, Y, Z | Linear acceleration (m/s²) |
| Gyroscope | X, Y, Z | Angular velocity (rad/s) |
| Magnetometer | X, Y, Z | Magnetic field (μT) |

**Sensor Fusion:**

IMUs typically combine all sensor data to estimate orientation:

```
Orientation = integrate(gyroscope) + correct_with(accelerometer + magnetometer)
```

**Common IMU Models:**
- **MPU-6050/9250**: Low-cost, hobbyist projects
- **BNO055**: Built-in sensor fusion
- **Bosch BMI088**: High-performance automotive
- **LORD MicroStrain**: Industrial-grade

**Applications:**
- Balancing for bipedal robots
- Drone stabilization
- Gait analysis
- Vehicle dynamics

```python
# Example: Reading IMU data with ROS 2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import math

class IMUMonitor(Node):
    def __init__(self):
        super().__init__('imu_monitor')
        self.subscription = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10)
    
    def imu_callback(self, msg):
        # Extract orientation (quaternion)
        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z
        qw = msg.orientation.w
        
        # Convert to Euler angles (roll, pitch, yaw)
        roll = math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        pitch = math.asin(2*(qw*qy - qz*qx))
        yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
        
        # Extract linear acceleration
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z
        
        self.get_logger().info(
            f'Roll: {math.degrees(roll):.1f}° '
            f'Pitch: {math.degrees(pitch):.1f}° '
            f'Yaw: {math.degrees(yaw):.1f}°')
```

### Wheel Encoders

Measure wheel rotation to estimate robot position and velocity.

**Types:**
- **Optical**: LED and photodetector
- **Magnetic**: Hall effect sensors
- **Capacitive**: Capacitance changes

**Key Specifications:**
- **Resolution**: Pulses per revolution (PPR)
  - Low: 12-64 PPR
  - Medium: 100-512 PPR
  - High: 1024-4096 PPR
- **Accuracy**: ±0.1° to ±0.01°

**Odometry Calculation:**

```python
# Simple differential drive odometry
class Odometry:
    def __init__(self, wheel_radius, wheel_base):
        self.r = wheel_radius  # meters
        self.L = wheel_base    # meters
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
    
    def update(self, left_ticks, right_ticks, ticks_per_rev):
        # Convert ticks to distance
        left_dist = (left_ticks / ticks_per_rev) * 2 * 3.14159 * self.r
        right_dist = (right_ticks / ticks_per_rev) * 2 * 3.14159 * self.r
        
        # Calculate center distance and rotation
        center_dist = (left_dist + right_dist) / 2
        d_theta = (right_dist - left_dist) / self.L
        
        # Update pose
        self.x += center_dist * math.cos(self.theta + d_theta/2)
        self.y += center_dist * math.sin(self.theta + d_theta/2)
        self.theta += d_theta
```

---

## Force and Tactile Sensors {#force-sensors}

### Force/Torque Sensors

Measure forces and moments applied to a robot, crucial for manipulation and contact interactions.

**6-DOF Force/Torque Sensor:**

| Measurement | Symbol | Unit |
|-------------|--------|------|
| Force X | Fx | Newton (N) |
| Force Y | Fy | Newton (N) |
| Force Z | Fz | Newton (N) |
| Torque X | Tx | Newton-meter (Nm) |
| Torque Y | Ty | Newton-meter (Nm) |
| Torque Z | Tz | Newton-meter (Nm) |

**Applications:**
- Precise object grasping
- Assembly operations (peg-in-hole)
- Human-robot collaboration (safety)
- Walking robots (ground contact force)
- Surgical robots (tissue interaction)

**Example: ATI Mini40**
- Range: ±40N (Fx, Fy), ±120N (Fz)
- Resolution: 1/50 N
- Compact size (40mm diameter)

### Tactile Sensors

Provide information about contact, pressure distribution, and texture.

**Types:**

1. **Resistive**: Pressure changes resistance
2. **Capacitive**: Pressure changes capacitance
3. **Piezoelectric**: Pressure generates voltage
4. **Optical**: Pressure deforms optical waveguides

**Applications:**
- Robotic grippers (grasp stability)
- Prosthetic hands (feedback)
- Texture recognition
- Slip detection

**Example: BioTac:**
- Mimics human fingertip
- 19 sensors (pressure, temperature, vibration)
- Can distinguish 117 materials with 95% accuracy

---

## Sensor Fusion {#sensor-fusion}

Single sensors have limitations. Sensor fusion combines data from multiple sensors to create a more accurate and robust perception.

### Why Sensor Fusion?

**Individual Sensor Limitations:**

| Sensor | Strengths | Weaknesses |
|--------|-----------|------------|
| Camera | Rich detail, color | No depth, light-dependent |
| LiDAR | Accurate 3D | Expensive, no color |
| Radar | All-weather, velocity | Low resolution |
| IMU | High-rate motion | Drift over time |
| GPS | Absolute position | Poor indoors, low rate |

**Fusion Benefits:**
- ✅ Compensates for individual weaknesses
- ✅ Increases reliability
- ✅ Provides redundancy
- ✅ Improves accuracy

### Fusion Techniques

**1. Kalman Filter**

Optimal for linear systems with Gaussian noise.

```
Prediction:    x̂ₖ = Aₖxₖ₋₁ + Bₖuₖ
Update:        xₖ = x̂ₖ + Kₖ(zₖ - Hx̂ₖ)
```

**2. Extended Kalman Filter (EKF)**

Handles nonlinear systems (e.g., robot orientation).

**3. Particle Filter**

Uses multiple "particles" to represent probability distribution.

**4. Deep Learning**

Neural networks learn optimal fusion from data.

### Example: Camera + LiDAR Fusion

```python
# Simplified camera-LiDAR fusion for object detection
class SensorFusion:
    def __init__(self):
        self.camera_detections = []
        self.lidar_points = []
    
    def fuse(self):
        fused_objects = []
        
        for detection in self.camera_detections:
            # Get 2D bounding box from camera
            x1, y1, x2, y2 = detection['bbox']
            object_class = detection['class']
            
            # Project LiDAR points into camera frame
            points_in_box = self.get_lidar_in_bbox(x1, y1, x2, y2)
            
            # Calculate 3D position from LiDAR
            if len(points_in_box) > 0:
                position_3d = np.mean(points_in_box, axis=0)
                
                fused_objects.append({
                    'class': object_class,
                    'position': position_3d,
                    'confidence': detection['confidence']
                })
        
        return fused_objects
```

---

## Perception Pipeline {#perception-pipeline}

A complete perception system processes raw sensor data through multiple stages to extract meaningful information.

### Standard Pipeline:

```
1. Data Acquisition
   ↓
2. Preprocessing
   ↓
3. Feature Extraction
   ↓
4. Object Detection/Recognition
   ↓
5. Tracking
   ↓
6. Scene Understanding
   ↓
7. Decision Making
```

### Stage Details:

**1. Data Acquisition**
- Read sensor data at appropriate rates
- Synchronize multi-sensor data
- Buffer for processing

**2. Preprocessing**
- Noise filtering
- Calibration correction
- Data format conversion

**3. Feature Extraction**
- Edges, corners, keypoints (vision)
- Planar surfaces (LiDAR)
- Motion patterns (IMU)

**4. Detection/Recognition**
- Object detection (YOLO, Faster R-CNN)
- Semantic segmentation
- Point cloud clustering

**5. Tracking**
- Maintain object identity over time
- Predict future positions
- Handle occlusions

**6. Scene Understanding**
- Build occupancy map
- Classify drivable surfaces
- Understand spatial relationships

**7. Decision Making**
- Path planning
- Behavior selection
- Control commands

---

## Challenges in Real-World Perception {#challenges}

### Environmental Challenges

**Lighting Variations:**
- Bright sunlight causes glare and washout
- Darkness requires active illumination
- Shadows create false obstacles

**Weather Conditions:**
- Rain/snow affects optical sensors
- Fog reduces visibility range
- Ice/mud affects traction sensing

**Dynamic Environments:**
- Moving obstacles (pedestrians, vehicles)
- Changing layouts
- Temporary obstructions

### Technical Challenges

**Computational Limits:**
- Real-time processing requirements
- Limited onboard computing
- Power constraints

**Calibration Drift:**
- Sensors shift over time
- Temperature effects
- Mechanical wear

**Sensor Failures:**
- Hardware malfunctions
- Blocked sensors
- Data corruption

### Solutions:

| Challenge | Solution |
|-----------|----------|
| **Lighting** | Multi-modal sensing, HDR cameras |
| **Weather** | Radar + LiDAR fusion, thermal imaging |
| **Computation** | Edge AI accelerators, efficient algorithms |
| **Failures** | Redundancy, fault detection, graceful degradation |

---

## Practical Applications {#applications}

### Autonomous Vehicles

**Sensor Suite:**
- 📷 8+ cameras (360° coverage)
- 🔴 5+ LiDAR units
- 📡 6+ radar sensors
- 🧭 High-precision GPS + IMU
- 🌐 V2X communication

**Perception Tasks:**
- Lane detection
- Traffic sign recognition
- Pedestrian detection
- Vehicle tracking
- Free space estimation

### Warehouse Robots

**Sensors:**
- 2D LiDAR for navigation
- Depth camera for shelf detection
- Encoders for odometry
- Bump sensors for safety

**Tasks:**
- Autonomous navigation
- Pallet detection
- Collision avoidance
- Localization

### Humanoid Robots

**Sensor Requirements:**
- Head-mounted cameras
- Torso LiDAR/cameras
- IMU for balance
- Force sensors in feet
- Tactile sensors in hands

**Perception Needs:**
- Human detection and tracking
- Object recognition and grasping
- Terrain classification
- Balance maintenance

---

## Summary

Robot perception is the foundation of intelligent physical AI systems. By combining multiple sensor modalities—vision, depth, motion, and touch—robots can build comprehensive models of their environment and interact safely and effectively with the physical world.

**Key Takeaways:**

1. ✅ Different sensors provide complementary information
2. ✅ No single sensor is perfect—fusion is essential
3. ✅ Real-time processing is critical for robot control
4. ✅ Environmental challenges require robust solutions
5. ✅ Perception enables all higher-level robot behaviors

**Next Steps:**

In the following chapters, we'll explore how robots use this perceptual information to make decisions, plan actions, and execute complex tasks in the physical world.

---

*This chapter provided the foundation for understanding robot sensors and perception. The next chapter will dive into ROS 2 architecture, the middleware that connects sensors, algorithms, and actuators into complete robotic systems.*