# Chapter 9: Isaac ROS & Perception

> "Hardware-accelerated AI perception brings real-time intelligence to edge robots, from Jetson to data centers."

## Table of Contents

1. [Introduction to Isaac ROS](#introduction)
2. [Architecture Overview](#architecture)
3. [Installation and Setup](#installation)
4. [Visual SLAM](#vslam)
5. [Object Detection](#object-detection)
6. [Depth Processing](#depth)
7. [Navigation Stack](#navigation)
8. [Performance Optimization](#optimization)
9. [Deployment on Jetson](#jetson)

---

## Introduction to Isaac ROS {#introduction}

**Isaac ROS** is NVIDIA's collection of hardware-accelerated ROS 2 packages optimized for Jetson and NVIDIA GPUs.

### Why Isaac ROS?

| Traditional ROS | Isaac ROS |
|-----------------|-----------|
| CPU-based | GPU-accelerated |
| ~5-10 FPS | 30-60+ FPS |
| High latency | Ultra-low latency |
| Limited scalability | Massively parallel |

**Key Components:**

- **isaac_ros_visual_slam**: Real-time SLAM
- **isaac_ros_dnn_inference**: AI inference
- **isaac_ros_image_processing**: Image pipeline
- **isaac_ros_depth**: Depth estimation
- **isaac_ros_apriltag**: AprilTag detection
- **isaac_ros_nvblox**: 3D reconstruction

---

## Architecture Overview {#architecture}

### Isaac ROS Stack

```
┌─────────────────────────────────────┐
│        Application Layer            │
│    (Navigation, Manipulation)       │
├─────────────────────────────────────┤
│       Isaac ROS Packages            │
│  ┌──────────┐  ┌────────────────┐  │
│  │  VSLAM   │  │ Object Detection│ │
│  └──────────┘  └────────────────┘  │
│  ┌──────────┐  ┌────────────────┐  │
│  │  Depth   │  │   AprilTag     │  │
│  └──────────┘  └────────────────┘  │
├─────────────────────────────────────┤
│        NVIDIA Accelerators          │
│  ┌──────────┐  ┌────────────────┐  │
│  │ CUDA/DLA │  │  TensorRT      │  │
│  └──────────┘  └────────────────┘  │
│  ┌──────────┐  ┌────────────────┐  │
│  │   VPI    │  │    Triton      │  │
│  └──────────┘  └────────────────┘  │
└─────────────────────────────────────┘
```

### Hardware Acceleration Technologies

| Technology | Purpose |
|------------|---------|
| **CUDA** | Parallel GPU computing |
| **TensorRT** | Optimized DNN inference |
| **VPI (Vision Programming Interface)** | Accelerated image processing |
| **DLA (Deep Learning Accelerator)** | Dedicated AI engine (Jetson) |
| **Triton** | Inference serving |

---

## Installation and Setup {#installation}

### Prerequisites

```bash
# Install ROS 2 Humble
# (See Chapter 3 for details)

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Isaac ROS Installation

```bash
# Create workspace
mkdir -p ~/workspaces/isaac_ros-dev/src
cd ~/workspaces/isaac_ros-dev/src

# Clone Isaac ROS packages
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline.git

# Build using Docker
cd ~/workspaces/isaac_ros-dev
./src/isaac_ros_common/scripts/run_dev.sh

# Inside container, build
cd /workspaces/isaac_ros-dev
colcon build --symlink-install
source install/setup.bash
```

### Verify Installation

```bash
# Test CUDA
nvidia-smi

# Test Isaac ROS
ros2 pkg list | grep isaac_ros
```

---

## Visual SLAM {#vslam}

### cuVSLAM Overview

Isaac ROS Visual SLAM uses **cuVSLAM** (CUDA Visual SLAM) for real-time localization and mapping.

**Features:**
- ⚡ GPU-accelerated
- 🗺️ Real-time mapping
- 📍 6-DOF pose estimation
- 🎥 Stereo or mono camera support
- 🔄 Loop closure detection

### Basic VSLAM Launch

```python
# vslam.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    
    camera_type = LaunchConfiguration('camera_type')
    
    declare_camera_type = DeclareLaunchArgument(
        'camera_type',
        default_value='realsense',
        description='Camera type: realsense, zed, oak'
    )
    
    # Visual SLAM node
    visual_slam_node = Node(
        package='isaac_ros_visual_slam',
        executable='isaac_ros_visual_slam',
        name='visual_slam',
        parameters=[{
            'enable_rectified_pose': True,
            'denoise_input_images': False,
            'rectified_images': True,
            'enable_debug_mode': False,
            'debug_dump_path': '/tmp/cuvslam',
            'enable_slam_visualization': True,
            'enable_landmarks_view': True,
            'enable_observations_view': True,
            'map_frame': 'map',
            'odom_frame': 'odom',
            'base_frame': 'base_link',
            'input_camera_frame': 'camera_infra1_optical_frame',
            'enable_localization_n_mapping': True,
            'path_max_size': 1024,
        }],
        remappings=[
            ('stereo_camera/left/image', '/camera/infra1/image_rect_raw'),
            ('stereo_camera/left/camera_info', '/camera/infra1/camera_info'),
            ('stereo_camera/right/image', '/camera/infra2/image_rect_raw'),
            ('stereo_camera/right/camera_info', '/camera/infra2/camera_info'),
        ],
        output='screen'
    )
    
    return LaunchDescription([
        declare_camera_type,
        visual_slam_node
    ])
```

### VSLAM with RealSense

```python
# realsense_vslam.launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    
    # RealSense camera
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('realsense2_camera'),
            '/launch/rs_launch.py'
        ]),
        launch_arguments={
            'enable_infra1': 'true',
            'enable_infra2': 'true',
            'enable_depth': 'true',
            'depth_module.profile': '640x480x30',
            'enable_gyro': 'true',
            'enable_accel': 'true',
            'gyro_fps': '200',
            'accel_fps': '200',
            'unite_imu_method': '1'
        }.items()
    )
    
    # Visual SLAM
    vslam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('isaac_ros_visual_slam'),
            '/launch/isaac_ros_visual_slam_realsense.launch.py'
        ])
    )
    
    return LaunchDescription([
        realsense_launch,
        vslam_launch
    ])
```

### Monitoring VSLAM

```bash
# View pose
ros2 topic echo /visual_slam/tracking/odometry

# View landmarks
ros2 topic echo /visual_slam/tracking/slam_path

# Visualize in RViz
rviz2 -d isaac_ros_visual_slam/rviz/default.rviz
```

---

## Object Detection {#object-detection}

### TensorRT Accelerated Detection

```python
# object_detection.launch.py
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    
    container = ComposableNodeContainer(
        name='detection_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            
            # Image rectification
            ComposableNode(
                package='isaac_ros_image_proc',
                plugin='nvidia::isaac_ros::image_proc::RectifyNode',
                name='rectify_node',
                remappings=[
                    ('image_raw', '/camera/image_raw'),
                    ('camera_info', '/camera/camera_info'),
                    ('image_rect', '/image_rect'),
                ]
            ),
            
            # DNN inference
            ComposableNode(
                package='isaac_ros_dnn_inference',
                plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
                name='tensorrt_node',
                parameters=[{
                    'model_file_path': '/tmp/models/yolov5s.onnx',
                    'engine_file_path': '/tmp/models/yolov5s.engine',
                    'input_tensor_names': ['input'],
                    'input_binding_names': ['input'],
                    'output_tensor_names': ['output'],
                    'output_binding_names': ['output'],
                    'verbose': False,
                }],
                remappings=[
                    ('tensor_pub', '/tensor'),
                    ('tensor_sub', '/image_tensor'),
                ]
            ),
            
            # Detection decoder
            ComposableNode(
                package='isaac_ros_yolov5',
                plugin='nvidia::isaac_ros::yolov5::YoloV5DecoderNode',
                name='yolov5_decoder',
                parameters=[{
                    'confidence_threshold': 0.5,
                    'nms_threshold': 0.45,
                }],
                remappings=[
                    ('tensor', '/tensor'),
                    ('detections', '/detections'),
                ]
            ),
        ],
        output='screen'
    )
    
    return LaunchDescription([container])
```

### Custom Model Integration

```python
import torch
import tensorrt as trt

def convert_pytorch_to_tensorrt(pytorch_model, input_shape):
    """Convert PyTorch model to TensorRT"""
    
    # Export to ONNX
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        'model.onnx',
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}}
    )
    
    # Build TensorRT engine
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    
    with open('model.onnx', 'rb') as f:
        parser.parse(f.read())
    
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16
    
    engine = builder.build_engine(network, config)
    
    # Save engine
    with open('model.engine', 'wb') as f:
        f.write(engine.serialize())
    
    return 'model.engine'
```

---

## Depth Processing {#depth}

### Stereo Depth Estimation

```python
# stereo_depth.launch.py
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    
    container = ComposableNodeContainer(
        name='stereo_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            
            # Disparity calculation
            ComposableNode(
                package='isaac_ros_stereo_image_proc',
                plugin='nvidia::isaac_ros::stereo_image_proc::DisparityNode',
                name='disparity',
                parameters=[{
                    'window_size': 5,
                    'min_disparity': 0,
                    'max_disparity': 64,
                }],
                remappings=[
                    ('left/image_rect', '/left/image_rect'),
                    ('left/camera_info', '/left/camera_info'),
                    ('right/image_rect', '/right/image_rect'),
                    ('right/camera_info', '/right/camera_info'),
                ]
            ),
            
            # Point cloud generation
            ComposableNode(
                package='isaac_ros_stereo_image_proc',
                plugin='nvidia::isaac_ros::stereo_image_proc::PointCloudNode',
                name='point_cloud',
                parameters=[{
                    'use_color': True,
                    'unit_scaling': 1.0,
                }],
                remappings=[
                    ('left/image_rect_color', '/left/image_rect_color'),
                    ('disparity', '/disparity'),
                ]
            ),
        ],
        output='screen'
    )
    
    return LaunchDescription([container])
```

### nvblox for 3D Reconstruction

```python
# nvblox.launch.py
from launch_ros.actions import Node

def generate_launch_description():
    
    nvblox_node = Node(
        package='nvblox_ros',
        executable='nvblox_node',
        name='nvblox_node',
        parameters=[{
            'voxel_size': 0.05,
            'esdf_mode': 2,  # 3D ESDF
            'esdf_2d_min_height': 0.0,
            'esdf_2d_max_height': 2.0,
            'max_integration_distance_m': 10.0,
            'max_tsdf_update_hz': 10.0,
            'max_color_update_hz': 5.0,
            'max_mesh_update_hz': 5.0,
            'max_esdf_update_hz': 2.0,
            'mesh_bandwidth_limit_mbps': 20.0,
        }],
        remappings=[
            ('depth/image', '/camera/depth/image_rect_raw'),
            ('depth/camera_info', '/camera/depth/camera_info'),
            ('color/image', '/camera/color/image_raw'),
            ('color/camera_info', '/camera/color/camera_info'),
            ('pose', '/visual_slam/tracking/odometry'),
        ],
        output='screen'
    )
    
    return LaunchDescription([nvblox_node])
```

---

## Navigation Stack {#navigation}

### Isaac ROS Nav2 Integration

```python
# nav2_isaac.launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    # Nav2
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('nav2_bringup'),
            '/launch/navigation_launch.py'
        ]),
        launch_arguments={
            'params_file': 'nav2_params.yaml',
            'use_sim_time': 'false'
        }.items()
    )
    
    # Costmap layers with Isaac ROS
    costmap_params = {
        'obstacle_layer': {
            'plugin': 'isaac_ros_costmap_2d::ObstacleLayer',
            'observation_sources': 'scan nvblox',
            'scan': {
                'topic': '/scan',
                'sensor_frame': 'lidar_link',
                'observation_persistence': 0.0,
                'expected_update_rate': 10.0,
            },
            'nvblox': {
                'topic': '/nvblox_node/map_slice',
                'sensor_frame': 'camera_link',
                'observation_persistence': 0.0,
                'expected_update_rate': 5.0,
            }
        }
    }
    
    return LaunchDescription([
        nav2_launch
    ])
```

---

## Performance Optimization {#optimization}

### Benchmarking

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import time

class PerformanceMonitor(Node):
    def __init__(self):
        super().__init__('performance_monitor')
        
        self.subscription = self.create_subscription(
            Image,
            '/detections_image',
            self.callback,
            10
        )
        
        self.frame_times = []
        self.last_time = time.time()
    
    def callback(self, msg):
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.frame_times.append(frame_time)
        
        if len(self.frame_times) >= 100:
            avg_fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
            self.get_logger().info(f'Average FPS: {avg_fps:.2f}')
            self.frame_times.clear()
        
        self.last_time = current_time
```

### Optimization Strategies

| Technique | Impact | Notes |
|-----------|--------|-------|
| **Use DLA on Jetson** | High | Offload to dedicated AI engine |
| **FP16 inference** | Medium | 2x speedup, minimal accuracy loss |
| **INT8 quantization** | High | 4x speedup, requires calibration |
| **Batch processing** | High | Process multiple frames together |
| **Zero-copy** | Medium | Avoid unnecessary data copies |
| **Multi-stream** | High | Parallel GPU operations |

### Multi-Stream Processing

```python
import torch

class MultiStreamProcessor:
    def __init__(self):
        self.streams = [torch.cuda.Stream() for _ in range(4)]
        self.current_stream = 0
    
    def process_batch(self, images):
        stream = self.streams[self.current_stream]
        
        with torch.cuda.stream(stream):
            # GPU operations here
            processed = self.model(images)
        
        self.current_stream = (self.current_stream + 1) % len(self.streams)
        
        return processed
```

---

## Deployment on Jetson {#jetson}

### Jetson Setup

```bash
# Flash Jetson with JetPack 5.1+
# Use NVIDIA SDK Manager

# Install Isaac ROS on Jetson
sudo apt-get update
sudo apt-get install nvidia-jetpack

# Set power mode
sudo nvpmodel -m 0  # Max performance
sudo jetson_clocks  # Max clocks
```

### Jetson-Optimized Launch

```python
# jetson_perception.launch.py
def generate_launch_description():
    
    # Use DLA for inference
    tensorrt_node = ComposableNode(
        package='isaac_ros_dnn_inference',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        name='tensorrt_dla',
        parameters=[{
            'model_file_path': '/models/model.onnx',
            'engine_file_path': '/models/model_dla.engine',
            'dla_core': 0,  # Use DLA core 0
            'enable_fp16': True,
        }]
    )
    
    return LaunchDescription([
        ComposableNodeContainer(
            name='perception_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[tensorrt_node],
            output='screen'
        )
    ])
```

### Monitoring Jetson

```bash
# Monitor resources
tegrastats

# Check GPU utilization
nvidia-smi

# Profile application
nsys profile -o profile.qdrep ros2 launch my_package perception.launch.py
```

---

## Summary

Isaac ROS brings hardware-accelerated AI perception to ROS 2, enabling real-time intelligent behaviors on edge devices and in data centers.

**Key Takeaways:**

1. ✅ GPU-accelerated perception (10-30x speedup)
2. ✅ Real-time VSLAM with cuVSLAM
3. ✅ TensorRT optimized DNN inference
4. ✅ Hardware acceleration on Jetson
5. ✅ Seamless ROS 2 integration
6. ✅ Production-ready performance

**Next Chapter:**

Chapter 10 explores navigation and reinforcement learning for autonomous mobile robots.

---

*This chapter introduced Isaac ROS for hardware-accelerated perception. You can now deploy real-time AI on edge robots with professional-grade performance.*