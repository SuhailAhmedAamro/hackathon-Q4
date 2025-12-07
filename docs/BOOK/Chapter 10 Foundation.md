# Chapter 10: Navigation & Reinforcement Learning

> "Autonomous navigation combines classical planning with learned behaviors—the best of both worlds."

## Table of Contents

1. [Introduction to Robot Navigation](#introduction)
2. [Nav2 Stack Overview](#nav2)
3. [Mapping and Localization](#mapping)
4. [Path Planning](#planning)
5. [Reinforcement Learning Basics](#rl-basics)
6. [Training Navigation Policies](#training)
7. [Bipedal Walking](#biped)
8. [Sim-to-Real Transfer](#sim2real)

---

## Introduction to Robot Navigation {#introduction}

Robot navigation is the ability to move from one location to another while avoiding obstacles and adapting to the environment.

### Navigation Components

```
┌─────────────────────────────────────┐
│    High-Level Path Planning         │
├─────────────────────────────────────┤
│    Local Obstacle Avoidance         │
├─────────────────────────────────────┤
│         Localization                │
├─────────────────────────────────────┤
│           Mapping                   │
├─────────────────────────────────────┤
│         Perception                  │
└─────────────────────────────────────┘
```

---

## Nav2 Stack Overview {#nav2}

### Nav2 Architecture

Nav2 (Navigation2) is the complete autonomous navigation framework for ROS 2.

**Core Components:**

| Component | Function |
|-----------|----------|
| **Map Server** | Loads and serves maps |
| **AMCL** | Monte Carlo localization |
| **Planner** | Global path planning |
| **Controller** | Local trajectory tracking |
| **Recoveries** | Stuck situation handling |
| **Behavior Tree** | Mission coordination |

### Installing Nav2

```bash
sudo apt install ros-humble-navigation2 \
                 ros-humble-nav2-bringup \
                 ros-humble-slam-toolbox
```

### Basic Nav2 Launch

```python
# nav2_bringup.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_dir = get_package_share_directory('my_robot_nav')
    
    nav2_params = os.path.join(pkg_dir, 'config', 'nav2_params.yaml')
    map_file = os.path.join(pkg_dir, 'maps', 'office.yaml')
    
    return LaunchDescription([
        # Map server
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{'yaml_filename': map_file}]
        ),
        
        # AMCL
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[nav2_params]
        ),
        
        # Controller
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[nav2_params]
        ),
        
        # Planner
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[nav2_params]
        ),
        
        # Behavior server
        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            output='screen',
            parameters=[nav2_params]
        ),
        
        # BT Navigator
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[nav2_params]
        ),
        
        # Lifecycle manager
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{
                'autostart': True,
                'node_names': [
                    'map_server',
                    'amcl',
                    'controller_server',
                    'planner_server',
                    'behavior_server',
                    'bt_navigator'
                ]
            }]
        )
    ])
```

---

## Mapping and Localization {#mapping}

### SLAM Toolbox

```bash
# Run SLAM
ros2 launch slam_toolbox online_async_launch.py

# Save map
ros2 run nav2_map_server map_saver_cli -f my_map
```

### SLAM Configuration

```yaml
# slam_config.yaml
slam_toolbox:
  ros__parameters:
    odom_frame: odom
    map_frame: map
    base_frame: base_link
    scan_topic: /scan
    mode: mapping
    
    # Resolution
    resolution: 0.05
    
    # Range
    max_laser_range: 20.0
    minimum_travel_distance: 0.5
    minimum_travel_heading: 0.5
    
    # Loop closure
    loop_search_maximum_distance: 3.0
    do_loop_closing: true
    loop_match_minimum_chain_size: 10
```

### AMCL Localization

```yaml
# amcl_config.yaml
amcl:
  ros__parameters:
    use_sim_time: false
    
    # Particle filter
    min_particles: 500
    max_particles: 2000
    
    # Update rates
    update_min_d: 0.2
    update_min_a: 0.5
    
    # Odometry model
    odom_model_type: "diff-corrected"
    odom_alpha1: 0.2
    odom_alpha2: 0.2
    odom_alpha3: 0.2
    odom_alpha4: 0.2
    
    # Laser model
    laser_model_type: "likelihood_field"
    laser_likelihood_max_dist: 2.0
    laser_max_range: 20.0
    laser_min_range: 0.1
```

---

## Path Planning {#planning}

### Global Planner

```yaml
# planner_config.yaml
planner_server:
  ros__parameters:
    planner_plugins: ["GridBased"]
    
    GridBased:
      plugin: "nav2_navfn_planner/NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true
```

### Local Controller (DWB)

```yaml
# controller_config.yaml
controller_server:
  ros__parameters:
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.001
    min_theta_velocity_threshold: 0.001
    
    controller_plugins: ["FollowPath"]
    
    FollowPath:
      plugin: "dwb_core::DWBLocalPlanner"
      
      # Velocity limits
      min_vel_x: 0.0
      max_vel_x: 0.5
      max_vel_theta: 1.0
      min_speed_xy: 0.0
      max_speed_xy: 0.5
      
      # Acceleration limits
      acc_lim_x: 2.5
      acc_lim_theta: 3.2
      decel_lim_x: -2.5
      decel_lim_theta: -3.2
      
      # Trajectory generation
      vx_samples: 20
      vy_samples: 5
      vtheta_samples: 20
      sim_time: 1.7
      
      # Scoring
      critics: [
        "RotateToGoal",
        "Oscillation",
        "BaseObstacle",
        "GoalAlign",
        "PathAlign",
        "PathDist",
        "GoalDist"
      ]
```

### Costmap Configuration

```yaml
# costmap_common.yaml
costmap_common:
  ros__parameters:
    footprint: "[[0.3, 0.25], [0.3, -0.25], [-0.3, -0.25], [-0.3, 0.25]]"
    robot_radius: 0.4
    
    obstacle_layer:
      plugin: "nav2_costmap_2d::ObstacleLayer"
      enabled: True
      observation_sources: scan
      scan:
        topic: /scan
        max_obstacle_height: 2.0
        clearing: True
        marking: True
    
    inflation_layer:
      plugin: "nav2_costmap_2d::InflationLayer"
      cost_scaling_factor: 3.0
      inflation_radius: 0.55
    
    static_layer:
      plugin: "nav2_costmap_2d::StaticLayer"
      map_subscribe_transient_local: True
```

---

## Reinforcement Learning Basics {#rl-basics}

### RL for Robotics

Reinforcement Learning trains robots through trial and error using rewards.

**Key Concepts:**

| Concept | Description |
|---------|-------------|
| **Agent** | The robot |
| **Environment** | Simulation or real world |
| **State** | Robot's observations |
| **Action** | Robot's commands |
| **Reward** | Feedback signal |
| **Policy** | State→Action mapping |

### Markov Decision Process (MDP)

```
Agent observes State (s)
  ↓
Agent takes Action (a)
  ↓
Environment transitions to State (s')
  ↓
Agent receives Reward (r)
  ↓
Repeat
```

### Training Loop

```python
import gym
import torch
import torch.nn as nn

class RobotPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.network(x)

# Training
env = gym.make('RobotNavigation-v0')
policy = RobotPolicy(obs_dim=10, action_dim=2)
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

for episode in range(10000):
    obs = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        # Get action from policy
        obs_tensor = torch.FloatTensor(obs)
        action = policy(obs_tensor).detach().numpy()
        
        # Step environment
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
        
        # Update policy (simplified)
        loss = -reward  # Policy gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Episode {episode}: Reward = {episode_reward}")
```

---

## Training Navigation Policies {#training}

### Custom Navigation Environment

```python
import gym
from gym import spaces
import numpy as np

class NavigationEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Observation: [x, y, theta, goal_x, goal_y, lidar_10_rays]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(15,),
            dtype=np.float32
        )
        
        # Action: [linear_vel, angular_vel]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        # Random start and goal
        self.robot_pos = np.random.uniform(-5, 5, size=2)
        self.robot_theta = np.random.uniform(-np.pi, np.pi)
        self.goal_pos = np.random.uniform(-5, 5, size=2)
        
        # Initialize lidar
        self.lidar_readings = np.ones(10) * 10.0
        
        return self._get_obs()
    
    def step(self, action):
        # Apply action
        linear_vel, angular_vel = action
        dt = 0.1
        
        self.robot_theta += angular_vel * dt
        self.robot_pos[0] += linear_vel * np.cos(self.robot_theta) * dt
        self.robot_pos[1] += linear_vel * np.sin(self.robot_theta) * dt
        
        # Update lidar (simplified)
        self.lidar_readings = self._scan_lidar()
        
        # Calculate reward
        distance_to_goal = np.linalg.norm(self.robot_pos - self.goal_pos)
        
        reward = -distance_to_goal  # Closer to goal = higher reward
        reward -= 0.1 * np.abs(angular_vel)  # Penalize rotation
        
        # Check collision
        if np.min(self.lidar_readings) < 0.3:
            reward -= 10.0
            done = True
        # Check goal reached
        elif distance_to_goal < 0.5:
            reward += 100.0
            done = True
        else:
            done = False
        
        return self._get_obs(), reward, done, {}
    
    def _get_obs(self):
        return np.concatenate([
            self.robot_pos,
            [self.robot_theta],
            self.goal_pos,
            self.lidar_readings
        ])
    
    def _scan_lidar(self):
        # Simplified lidar simulation
        return np.random.uniform(0.3, 10.0, size=10)
```

### PPO Training

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# Create parallel environments
def make_env():
    def _init():
        return NavigationEnv()
    return _init

num_envs = 8
env = SubprocVecEnv([make_env() for _ in range(num_envs)])

# Create PPO agent
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    tensorboard_log="./tensorboard_logs/"
)

# Train
model.learn(total_timesteps=1_000_000)

# Save
model.save("navigation_policy")
```

### Deploying Learned Policy

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import torch
import numpy as np

class LearnedNavigationNode(Node):
    def __init__(self, policy_path):
        super().__init__('learned_navigation')
        
        # Load policy
        self.policy = torch.load(policy_path)
        self.policy.eval()
        
        # Publishers and subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        
        self.latest_scan = None
        self.timer = self.create_timer(0.1, self.control_loop)
    
    def scan_callback(self, msg):
        self.latest_scan = msg
    
    def control_loop(self):
        if self.latest_scan is None:
            return
        
        # Prepare observation
        obs = self._get_observation()
        
        # Get action from policy
        with torch.no_grad():
            action = self.policy(torch.FloatTensor(obs)).numpy()
        
        # Publish command
        cmd = Twist()
        cmd.linear.x = float(action[0])
        cmd.angular.z = float(action[1])
        self.cmd_pub.publish(cmd)
    
    def _get_observation(self):
        # Extract features from scan
        lidar = np.array(self.latest_scan.ranges[::36])  # 10 rays
        # Add other observations (position, goal, etc.)
        return np.concatenate([lidar, [0, 0, 0, 1, 1]])  # Simplified
```

---

## Bipedal Walking {#biped}

### ZMP Control

Zero Moment Point (ZMP) is critical for bipedal stability.

```python
class ZMPController:
    def __init__(self, robot_mass, com_height):
        self.mass = robot_mass
        self.h = com_height
        self.g = 9.81
    
    def compute_zmp(self, com_pos, com_acc):
        """
        ZMP = CoM - (CoM_height / g) * CoM_acceleration_xy
        """
        zmp_x = com_pos[0] - (self.h / self.g) * com_acc[0]
        zmp_y = com_pos[1] - (self.h / self.g) * com_acc[1]
        return np.array([zmp_x, zmp_y])
    
    def is_stable(self, zmp, support_polygon):
        """Check if ZMP is inside support polygon"""
        from shapely.geometry import Point, Polygon
        
        point = Point(zmp[0], zmp[1])
        polygon = Polygon(support_polygon)
        
        return polygon.contains(point)
```

### Gait Generation

```python
class GaitGenerator:
    def __init__(self, step_length, step_height, step_duration):
        self.step_length = step_length
        self.step_height = step_height
        self.step_duration = step_duration
    
    def generate_trajectory(self, t):
        """Generate foot trajectory for time t"""
        phase = (t % self.step_duration) / self.step_duration
        
        if phase < 0.5:
            # Swing phase
            x = self.step_length * (phase * 2)
            z = self.step_height * np.sin(phase * 2 * np.pi)
        else:
            # Stance phase
            x = self.step_length
            z = 0
        
        return np.array([x, 0, z])
```

### Bipedal RL Training

```python
class BipedalWalkingEnv(gym.Env):
    def __init__(self):
        # Observation: joint angles, velocities, orientation
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(24,),
            dtype=np.float32
        )
        
        # Action: joint torques
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32
        )
    
    def step(self, action):
        # Apply torques to joints
        self.robot.set_joint_torques(action * 100)  # Scale
        
        # Step physics
        self.sim.step()
        
        # Calculate reward
        forward_velocity = self.robot.get_forward_velocity()
        height = self.robot.get_com_height()
        energy = np.sum(np.abs(action))
        
        reward = forward_velocity  # Encourage forward motion
        reward += 1.0 if height > 0.7 else -1.0  # Stay upright
        reward -= 0.1 * energy  # Energy efficiency
        
        # Check termination
        done = height < 0.5 or np.abs(self.robot.get_roll()) > 0.5
        
        return self._get_obs(), reward, done, {}
```

---

## Sim-to-Real Transfer {#sim2real}

### Domain Randomization

```python
class RandomizedEnv(gym.Env):
    def reset(self):
        # Randomize physics
        self.sim.set_gravity(np.random.uniform(9.5, 10.5))
        
        # Randomize masses
        for link in self.robot.links:
            mass = link.mass * np.random.uniform(0.8, 1.2)
            link.set_mass(mass)
        
        # Randomize friction
        for joint in self.robot.joints:
            friction = np.random.uniform(0.0, 0.5)
            joint.set_friction(friction)
        
        # Randomize actuator delays
        self.actuator_delay = np.random.uniform(0.0, 0.05)
        
        return self._get_obs()
```

### System Identification

```python
def measure_real_robot_parameters():
    """Collect data from real robot to refine sim params"""
    
    data = {
        'joint_positions': [],
        'joint_velocities': [],
        'joint_torques': [],
        'timestamps': []
    }
    
    # Collect data for 60 seconds
    for t in range(600):
        # Apply known torque
        torque = np.sin(2 * np.pi * t / 100)
        robot.set_joint_torque(torque)
        
        # Measure response
        data['joint_positions'].append(robot.get_joint_position())
        data['joint_velocities'].append(robot.get_joint_velocity())
        data['joint_torques'].append(torque)
        data['timestamps'].append(t * 0.1)
        
        time.sleep(0.1)
    
    # Fit model to data
    estimated_params = fit_dynamic_model(data)
    
    return estimated_params
```

---

## Summary

Navigation combines classical planning algorithms with modern learning approaches. Reinforcement learning enables robots to learn complex behaviors through experience.

**Key Takeaways:**

1. ✅ Nav2 for classical autonomous navigation
2. ✅ SLAM for mapping and localization
3. ✅ RL for learning navigation policies
4. ✅ ZMP for bipedal stability
5. ✅ Domain randomization for sim-to-real
6. ✅ Combine planning and learning

**Next Chapter:**

Chapter 11 covers humanoid kinematics, the mathematical foundation for controlling articulated robots.

---

*This chapter explored navigation and reinforcement learning for autonomous robots. You can now build intelligent navigation systems using both classical and learning-based approaches.*