# Chapter 11: Humanoid Kinematics

> "Kinematics is the language of robot motion—transforming joint angles into end-effector positions and vice versa."

## Table of Contents

1. [Introduction to Kinematics](#introduction)
2. [Forward Kinematics](#forward-kinematics)
3. [Inverse Kinematics](#inverse-kinematics)
4. [Jacobian and Velocity Kinematics](#jacobian)
5. [Humanoid Arm Kinematics](#arm)
6. [Humanoid Leg Kinematics](#leg)
7. [Whole-Body Control](#whole-body)
8. [Practical Implementation](#implementation)

---

## Introduction to Kinematics {#introduction}

**Kinematics** studies robot motion without considering forces—the relationship between joint angles and end-effector positions.

### Key Concepts

| Term | Definition |
|------|------------|
| **Forward Kinematics (FK)** | Joint angles → End-effector pose |
| **Inverse Kinematics (IK)** | Desired pose → Joint angles |
| **Jacobian** | Velocity mapping between joint and task space |
| **Configuration Space** | Space of all possible joint angles |
| **Task Space** | Space of end-effector poses |

### Coordinate Frames

```
        Z (up)
        │
        │
        └───── Y
       ╱
      ╱
     X
```

### Homogeneous Transformations

```python
import numpy as np

def rotation_x(theta):
    """Rotation matrix around X axis"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c]
    ])

def rotation_y(theta):
    """Rotation matrix around Y axis"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])

def rotation_z(theta):
    """Rotation matrix around Z axis"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

def transform_matrix(rotation, translation):
    """Create 4x4 homogeneous transformation matrix"""
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T
```

---

## Forward Kinematics {#forward-kinematics}

### DH Parameters

Denavit-Hartenberg (DH) parameters describe robot geometry:

| Parameter | Description |
|-----------|-------------|
| **a** | Link length |
| **α** | Link twist |
| **d** | Link offset |
| **θ** | Joint angle |

### DH Transformation

```python
def dh_transform(a, alpha, d, theta):
    """Compute DH transformation matrix"""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    
    T = np.array([
        [ct,    -st*ca,  st*sa,   a*ct],
        [st,     ct*ca, -ct*sa,   a*st],
        [0,      sa,     ca,      d   ],
        [0,      0,      0,       1   ]
    ])
    
    return T
```

### Simple 3-DOF Arm Example

```python
class Arm3DOF:
    def __init__(self, l1=0.3, l2=0.3, l3=0.2):
        """3-DOF planar arm"""
        self.l1 = l1  # Link 1 length
        self.l2 = l2  # Link 2 length
        self.l3 = l3  # Link 3 length
    
    def forward_kinematics(self, q):
        """
        Compute end-effector position given joint angles
        q: [q1, q2, q3] (radians)
        Returns: [x, y, z, theta] (position and orientation)
        """
        q1, q2, q3 = q
        
        # Position
        x = (self.l1 * np.cos(q1) + 
             self.l2 * np.cos(q1 + q2) +
             self.l3 * np.cos(q1 + q2 + q3))
        
        y = (self.l1 * np.sin(q1) + 
             self.l2 * np.sin(q1 + q2) +
             self.l3 * np.sin(q1 + q2 + q3))
        
        z = 0  # Planar
        
        # Orientation
        theta = q1 + q2 + q3
        
        return np.array([x, y, z, theta])
    
    def compute_transform_matrix(self, q):
        """Compute full transformation matrix"""
        q1, q2, q3 = q
        
        # Transform from base to joint 1
        T01 = transform_matrix(
            rotation_z(q1),
            np.array([self.l1 * np.cos(q1), self.l1 * np.sin(q1), 0])
        )
        
        # Transform from joint 1 to joint 2
        T12 = transform_matrix(
            rotation_z(q2),
            np.array([self.l2 * np.cos(q2), self.l2 * np.sin(q2), 0])
        )
        
        # Transform from joint 2 to end-effector
        T23 = transform_matrix(
            rotation_z(q3),
            np.array([self.l3 * np.cos(q3), self.l3 * np.sin(q3), 0])
        )
        
        # Total transformation
        T03 = T01 @ T12 @ T23
        
        return T03

# Example
arm = Arm3DOF()
q = [0.5, 0.3, -0.2]  # Joint angles
ee_pose = arm.forward_kinematics(q)
print(f"End-effector position: {ee_pose}")
```

### 6-DOF Manipulator

```python
class Manipulator6DOF:
    def __init__(self):
        # DH parameters [a, alpha, d, theta_offset]
        self.dh_params = [
            [0,     np.pi/2,  0.3,  0],      # Joint 1
            [0.3,   0,        0,    0],      # Joint 2
            [0,     np.pi/2,  0,    0],      # Joint 3
            [0,    -np.pi/2,  0.3,  0],      # Joint 4
            [0,     np.pi/2,  0,    0],      # Joint 5
            [0,     0,        0.15, 0],      # Joint 6
        ]
    
    def forward_kinematics(self, joint_angles):
        """Compute FK for 6-DOF manipulator"""
        T = np.eye(4)
        
        for i, (a, alpha, d, theta_offset) in enumerate(self.dh_params):
            theta = joint_angles[i] + theta_offset
            Ti = dh_transform(a, alpha, d, theta)
            T = T @ Ti
        
        position = T[:3, 3]
        rotation = T[:3, :3]
        
        return position, rotation
```

---

## Inverse Kinematics {#inverse-kinematics}

### Analytical IK (2-Link Arm)

```python
def inverse_kinematics_2link(x, y, l1, l2):
    """
    Analytical IK for 2-link planar arm
    Returns: [q1, q2] or None if unreachable
    """
    # Check if target is reachable
    d = np.sqrt(x**2 + y**2)
    if d > l1 + l2 or d < abs(l1 - l2):
        return None  # Unreachable
    
    # Elbow up solution
    cos_q2 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_q2 = np.clip(cos_q2, -1, 1)  # Numerical safety
    
    q2 = np.arccos(cos_q2)
    
    k1 = l1 + l2 * np.cos(q2)
    k2 = l2 * np.sin(q2)
    
    q1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    
    return np.array([q1, q2])

# Example
l1, l2 = 0.3, 0.3
target = [0.4, 0.3]
joint_angles = inverse_kinematics_2link(*target, l1, l2)
print(f"Joint angles: {joint_angles}")
```

### Numerical IK (Jacobian-based)

```python
class NumericalIK:
    def __init__(self, robot, max_iterations=100, tolerance=1e-4):
        self.robot = robot
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def solve(self, target_pose, initial_guess):
        """
        Solve IK using iterative Jacobian method
        target_pose: Desired [x, y, z] position
        initial_guess: Initial joint angles
        """
        q = np.array(initial_guess)
        
        for iteration in range(self.max_iterations):
            # Compute current end-effector position
            current_pose = self.robot.forward_kinematics(q)[:3]
            
            # Compute error
            error = target_pose - current_pose
            
            # Check convergence
            if np.linalg.norm(error) < self.tolerance:
                return q, True
            
            # Compute Jacobian
            J = self.robot.compute_jacobian(q)
            
            # Compute joint velocity (damped least squares)
            lambda_damping = 0.01
            dq = J.T @ np.linalg.inv(J @ J.T + lambda_damping**2 * np.eye(3)) @ error
            
            # Update joint angles
            q += 0.5 * dq  # Step size
            
            # Joint limits (optional)
            q = np.clip(q, -np.pi, np.pi)
        
        return q, False  # Did not converge

# Example usage
robot = Arm3DOF()
ik_solver = NumericalIK(robot)
target = np.array([0.5, 0.3, 0])
initial_guess = [0, 0, 0]
solution, converged = ik_solver.solve(target, initial_guess)
print(f"Converged: {converged}, Solution: {solution}")
```

### CCD (Cyclic Coordinate Descent)

```python
def ccd_ik(robot, target, max_iterations=50):
    """Simple CCD IK solver"""
    q = np.zeros(robot.n_joints)
    
    for _ in range(max_iterations):
        # Work backwards from end-effector
        for i in range(robot.n_joints - 1, -1, -1):
            # Current end-effector position
            ee_pos = robot.forward_kinematics(q)[:3]
            
            # Joint position
            joint_pos = robot.get_joint_position(q, i)
            
            # Vectors
            to_end = ee_pos - joint_pos
            to_target = target - joint_pos
            
            # Compute rotation angle
            angle = np.arctan2(
                np.cross(to_end, to_target),
                np.dot(to_end, to_target)
            )
            
            # Update joint
            q[i] += angle
        
        # Check convergence
        error = np.linalg.norm(robot.forward_kinematics(q)[:3] - target)
        if error < 0.01:
            break
    
    return q
```

---

## Jacobian and Velocity Kinematics {#jacobian}

### Computing the Jacobian

```python
def compute_jacobian(robot, q, delta=1e-6):
    """
    Numerical Jacobian computation
    J[i,j] = ∂p_i/∂q_j
    """
    n_joints = len(q)
    ee_pos = robot.forward_kinematics(q)[:3]
    n_dims = len(ee_pos)
    
    J = np.zeros((n_dims, n_joints))
    
    for i in range(n_joints):
        q_plus = q.copy()
        q_plus[i] += delta
        
        ee_pos_plus = robot.forward_kinematics(q_plus)[:3]
        
        J[:, i] = (ee_pos_plus - ee_pos) / delta
    
    return J

# Analytical Jacobian for 3-DOF arm
class Arm3DOF:
    def compute_jacobian(self, q):
        """Analytical Jacobian"""
        q1, q2, q3 = q
        l1, l2, l3 = self.l1, self.l2, self.l3
        
        J = np.array([
            [
                -l1*np.sin(q1) - l2*np.sin(q1+q2) - l3*np.sin(q1+q2+q3),
                -l2*np.sin(q1+q2) - l3*np.sin(q1+q2+q3),
                -l3*np.sin(q1+q2+q3)
            ],
            [
                l1*np.cos(q1) + l2*np.cos(q1+q2) + l3*np.cos(q1+q2+q3),
                l2*np.cos(q1+q2) + l3*np.cos(q1+q2+q3),
                l3*np.cos(q1+q2+q3)
            ],
            [0, 0, 0]  # Planar, no Z motion
        ])
        
        return J
```

### Velocity Control

```python
def velocity_control(robot, target_velocity, current_q):
    """
    Compute joint velocities for desired end-effector velocity
    v = J * q_dot
    q_dot = J^{-1} * v (or pseudoinverse)
    """
    J = robot.compute_jacobian(current_q)
    
    # Moore-Penrose pseudoinverse
    J_pinv = np.linalg.pinv(J)
    
    # Compute joint velocities
    q_dot = J_pinv @ target_velocity
    
    return q_dot
```

---

## Humanoid Arm Kinematics {#arm}

### 7-DOF Humanoid Arm

```python
class HumanoidArm7DOF:
    def __init__(self):
        # Joint names
        self.joints = [
            'shoulder_pitch',
            'shoulder_roll',
            'shoulder_yaw',
            'elbow_pitch',
            'wrist_yaw',
            'wrist_pitch',
            'wrist_roll'
        ]
        
        # Link lengths
        self.upper_arm_length = 0.28
        self.forearm_length = 0.25
        self.hand_length = 0.15
    
    def forward_kinematics(self, q):
        """FK for 7-DOF arm"""
        # Shoulder transformations
        T_shoulder_pitch = transform_matrix(
            rotation_y(q[0]),
            np.zeros(3)
        )
        
        T_shoulder_roll = transform_matrix(
            rotation_x(q[1]),
            np.zeros(3)
        )
        
        T_shoulder_yaw = transform_matrix(
            rotation_z(q[2]),
            np.array([0, 0, self.upper_arm_length])
        )
        
        # Elbow
        T_elbow = transform_matrix(
            rotation_y(q[3]),
            np.array([0, 0, self.forearm_length])
        )
        
        # Wrist
        T_wrist_yaw = transform_matrix(
            rotation_z(q[4]),
            np.zeros(3)
        )
        
        T_wrist_pitch = transform_matrix(
            rotation_y(q[5]),
            np.zeros(3)
        )
        
        T_wrist_roll = transform_matrix(
            rotation_x(q[6]),
            np.array([0, 0, self.hand_length])
        )
        
        # Chain transformations
        T = (T_shoulder_pitch @ T_shoulder_roll @ T_shoulder_yaw @
             T_elbow @ T_wrist_yaw @ T_wrist_pitch @ T_wrist_roll)
        
        position = T[:3, 3]
        rotation = T[:3, :3]
        
        return position, rotation
```

---

## Humanoid Leg Kinematics {#leg}

### 6-DOF Leg

```python
class HumanoidLeg6DOF:
    def __init__(self):
        self.hip_offset = 0.1       # Hip width/2
        self.thigh_length = 0.4     # Thigh
        self.shin_length = 0.4      # Shin
        self.foot_height = 0.05     # Foot
    
    def forward_kinematics(self, q, side='left'):
        """
        FK for 6-DOF leg
        Joints: hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch, ankle_roll
        """
        hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch, ankle_roll = q
        
        # Hip offset (left vs right)
        offset = self.hip_offset if side == 'left' else -self.hip_offset
        
        # Hip transformations
        T_hip_yaw = transform_matrix(
            rotation_z(hip_yaw),
            np.array([0, offset, 0])
        )
        
        T_hip_roll = transform_matrix(
            rotation_x(hip_roll),
            np.zeros(3)
        )
        
        T_hip_pitch = transform_matrix(
            rotation_y(hip_pitch),
            np.array([0, 0, -self.thigh_length])
        )
        
        # Knee
        T_knee = transform_matrix(
            rotation_y(knee_pitch),
            np.array([0, 0, -self.shin_length])
        )
        
        # Ankle
        T_ankle_pitch = transform_matrix(
            rotation_y(ankle_pitch),
            np.zeros(3)
        )
        
        T_ankle_roll = transform_matrix(
            rotation_x(ankle_roll),
            np.array([0, 0, -self.foot_height])
        )
        
        # Total transformation
        T = (T_hip_yaw @ T_hip_roll @ T_hip_pitch @
             T_knee @ T_ankle_pitch @ T_ankle_roll)
        
        position = T[:3, 3]
        rotation = T[:3, :3]
        
        return position, rotation
    
    def inverse_kinematics_analytical(self, target_pos, target_rot):
        """Analytical IK for leg (simplified)"""
        x, y, z = target_pos
        
        # Compute leg length
        leg_extension = np.sqrt(x**2 + y**2 + z**2)
        
        # Knee angle (law of cosines)
        cos_knee = (leg_extension**2 - self.thigh_length**2 - self.shin_length**2) / \
                   (2 * self.thigh_length * self.shin_length)
        knee_pitch = np.arccos(np.clip(cos_knee, -1, 1))
        
        # Hip pitch
        alpha = np.arctan2(z, np.sqrt(x**2 + y**2))
        beta = np.arcsin((self.shin_length * np.sin(knee_pitch)) / leg_extension)
        hip_pitch = alpha + beta
        
        # Simplified: set other joints to zero
        q = np.array([0, 0, hip_pitch, knee_pitch, 0, 0])
        
        return q
```

---

## Whole-Body Control {#whole-body}

### Task Priority

```python
class WholeBodyController:
    def __init__(self, robot):
        self.robot = robot
    
    def compute_joint_velocities(self, tasks):
        """
        Compute joint velocities for multiple prioritized tasks
        tasks: List of (Jacobian, desired_velocity, priority)
        """
        n_joints = self.robot.n_joints
        q_dot = np.zeros(n_joints)
        N = np.eye(n_joints)  # Null space projector
        
        # Sort by priority
        tasks.sort(key=lambda x: x[2])
        
        for J, v_desired, priority in tasks:
            # Project into null space of higher priority tasks
            J_proj = J @ N
            
            # Compute pseudoinverse
            J_pinv = np.linalg.pinv(J_proj)
            
            # Compute task velocity
            q_dot += J_pinv @ (v_desired - J @ q_dot)
            
            # Update null space projector
            N = N @ (np.eye(n_joints) - J_pinv @ J_proj)
        
        return q_dot
```

---

## Practical Implementation {#implementation}

### ROS 2 IK Service

```python
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK

class IKServiceNode(Node):
    def __init__(self):
        super().__init__('ik_service')
        
        self.ik_client = self.create_client(
            GetPositionIK,
            '/compute_ik'
        )
    
    def compute_ik(self, target_pose):
        request = GetPositionIK.Request()
        request.ik_request.group_name = 'arm'
        request.ik_request.pose_stamped.pose = target_pose
        
        future = self.ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        response = future.result()
        if response.error_code.val == 1:
            return response.solution.joint_state
        else:
            return None
```

---

## Summary

Kinematics is fundamental to robot control. Forward kinematics maps joints to poses, inverse kinematics solves for desired configurations, and the Jacobian enables velocity control.

**Key Takeaways:**

1. ✅ Forward kinematics: joints → pose
2. ✅ Inverse kinematics: pose → joints
3. ✅ Jacobian for velocity mapping
4. ✅ Analytical IK when possible, numerical otherwise
5. ✅ Whole-body control for humanoids
6. ✅ ROS integration for real robots

**Next Chapter:**

Chapter 12 covers locomotion and manipulation—putting kinematics into action.

---

*This chapter explained humanoid kinematics. You now understand how to compute robot poses and solve for joint configurations to achieve desired motions.*