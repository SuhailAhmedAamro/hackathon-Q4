# Chapter 12: Locomotion & Manipulation

> "Walking and grasping—the hallmarks of embodied intelligence—require precise coordination of perception, planning, and control."

## Table of Contents

1. [Introduction](#introduction)
2. [Bipedal Walking Control](#walking)
3. [Balance and Stability](#balance)
4. [Grasping Fundamentals](#grasping)
5. [Manipulation Planning](#manipulation-planning)
6. [Force Control](#force-control)
7. [Integration: Walk and Manipulate](#integration)

---

## Introduction {#introduction}

Locomotion and manipulation are the two primary ways robots interact with their physical environment.

### Comparison

| Aspect | Locomotion | Manipulation |
|--------|------------|--------------|
| **Goal** | Move the robot | Move objects |
| **Challenges** | Balance, terrain | Precision, force |
| **Sensors** | IMU, force sensors | Cameras, tactile |
| **Control** | ZMP, CoM | IK, force control |

---

## Bipedal Walking Control {#walking}

### Gait Cycle

```
┌─────────────────────────────────┐
│      Single Support Phase       │  One foot on ground
├─────────────────────────────────┤
│      Double Support Phase       │  Both feet on ground
├─────────────────────────────────┤
│      Swing Phase                │  Moving foot
└─────────────────────────────────┘
```

### Walking Controller

```python
import numpy as np
from enum import Enum

class GaitPhase(Enum):
    LEFT_SUPPORT = 1
    DOUBLE_SUPPORT = 2
    RIGHT_SUPPORT = 3

class BipedalWalkingController:
    def __init__(self):
        self.step_length = 0.2      # meters
        self.step_height = 0.05     # meters
        self.step_duration = 0.8    # seconds
        self.double_support_ratio = 0.1
        
        self.phase = GaitPhase.LEFT_SUPPORT
        self.time_in_phase = 0
    
    def generate_foot_trajectory(self, t, phase):
        """Generate swing foot trajectory"""
        if phase == GaitPhase.LEFT_SUPPORT:
            # Right foot swings
            return self._swing_trajectory(t)
        elif phase == GaitPhase.RIGHT_SUPPORT:
            # Left foot swings
            return self._swing_trajectory(t)
        else:
            # Double support - no swing
            return np.array([0, 0, 0])
    
    def _swing_trajectory(self, t):
        """Foot trajectory during swing phase"""
        # Normalized time (0 to 1)
        tau = t / (self.step_duration * (1 - self.double_support_ratio))
        
        if tau > 1:
            tau = 1
        
        # Forward motion
        x = self.step_length * tau
        
        # Vertical motion (parabolic)
        z = 4 * self.step_height * tau * (1 - tau)
        
        # No lateral motion
        y = 0
        
        return np.array([x, y, z])
    
    def compute_com_trajectory(self, t):
        """Center of Mass trajectory for stability"""
        # Shift CoM over support foot
        com_offset = 0.05  # 5cm shift
        
        if self.phase == GaitPhase.LEFT_SUPPORT:
            com_y = com_offset
        elif self.phase == GaitPhase.RIGHT_SUPPORT:
            com_y = -com_offset
        else:
            com_y = 0
        
        # Smooth transition
        com_x = self.step_length * 0.5 * t / self.step_duration
        com_z = 0.8  # Constant height
        
        return np.array([com_x, com_y, com_z])
    
    def update(self, dt):
        """Update gait state"""
        self.time_in_phase += dt
        
        phase_duration = self.step_duration
        if self.phase == GaitPhase.DOUBLE_SUPPORT:
            phase_duration *= self.double_support_ratio
        
        if self.time_in_phase >= phase_duration:
            self.time_in_phase = 0
            self._transition_phase()
    
    def _transition_phase(self):
        """Transition to next gait phase"""
        if self.phase == GaitPhase.LEFT_SUPPORT:
            self.phase = GaitPhase.DOUBLE_SUPPORT
        elif self.phase == GaitPhase.DOUBLE_SUPPORT:
            # Alternate support leg
            if hasattr(self, '_last_support'):
                if self._last_support == GaitPhase.LEFT_SUPPORT:
                    self.phase = GaitPhase.RIGHT_SUPPORT
                else:
                    self.phase = GaitPhase.LEFT_SUPPORT
            else:
                self.phase = GaitPhase.RIGHT_SUPPORT
        else:
            self._last_support = self.phase
            self.phase = GaitPhase.DOUBLE_SUPPORT

# Example usage
controller = BipedalWalkingController()

for t in np.linspace(0, 2.0, 200):
    controller.update(0.01)
    foot_pos = controller.generate_foot_trajectory(t, controller.phase)
    com_pos = controller.compute_com_trajectory(t)
    print(f"t={t:.2f}, Phase={controller.phase}, Foot={foot_pos}, CoM={com_pos}")
```

### ZMP-Based Walking

```python
class ZMPWalkingController:
    def __init__(self):
        self.com_height = 0.8
        self.g = 9.81
        self.omega = np.sqrt(self.g / self.com_height)
    
    def compute_zmp(self, com_pos, com_vel, com_acc):
        """
        Compute Zero Moment Point
        ZMP = CoM - (h/g) * CoM_acc_xy
        """
        zmp_x = com_pos[0] - (self.com_height / self.g) * com_acc[0]
        zmp_y = com_pos[1] - (self.com_height / self.g) * com_acc[1]
        
        return np.array([zmp_x, zmp_y])
    
    def is_stable(self, zmp, support_polygon):
        """Check if ZMP is inside support polygon"""
        # Simple rectangular check
        min_x, max_x, min_y, max_y = support_polygon
        
        stable = (min_x <= zmp[0] <= max_x and 
                 min_y <= zmp[1] <= max_y)
        
        return stable
    
    def compute_com_reference(self, zmp_ref, t):
        """Compute CoM trajectory from ZMP reference"""
        # Simplified linear inverted pendulum model
        com_x = (zmp_ref[0] + 
                (self.com_height / self.g) * 
                self.omega**2 * np.sin(self.omega * t))
        
        com_y = (zmp_ref[1] + 
                (self.com_height / self.g) * 
                self.omega**2 * np.sin(self.omega * t))
        
        return np.array([com_x, com_y, self.com_height])
    
    def plan_zmp_trajectory(self, footsteps, step_duration):
        """Plan ZMP trajectory for sequence of footsteps"""
        zmp_trajectory = []
        
        for i, footstep in enumerate(footsteps):
            # During single support, ZMP is at support foot
            support_zmp = footstep[:2]  # x, y
            
            # Generate ZMP points
            num_points = int(step_duration / 0.01)
            for _ in range(num_points):
                zmp_trajectory.append(support_zmp)
        
        return np.array(zmp_trajectory)
```

---

## Balance and Stability {#balance}

### Inverted Pendulum Model

```python
class LinearInvertedPendulum:
    def __init__(self, mass, height):
        self.m = mass
        self.h = height
        self.g = 9.81
        self.omega = np.sqrt(self.g / self.h)
    
    def dynamics(self, state, zmp):
        """
        LIPM dynamics: ẍ = ω²(x - zmp)
        state: [x, ẋ]
        """
        x, x_dot = state
        x_ddot = self.omega**2 * (x - zmp)
        
        return np.array([x_dot, x_ddot])
    
    def simulate(self, initial_state, zmp_trajectory, dt):
        """Simulate LIPM"""
        states = [initial_state]
        
        for zmp in zmp_trajectory:
            # Euler integration
            state = states[-1]
            state_dot = self.dynamics(state, zmp)
            new_state = state + state_dot * dt
            states.append(new_state)
        
        return np.array(states)
```

### Capture Point Control

```python
class CapturePointController:
    def __init__(self, omega):
        self.omega = omega
    
    def compute_capture_point(self, com_pos, com_vel):
        """
        Capture point: position where robot must step 
        to come to rest
        ξ = x + ẋ/ω
        """
        xi_x = com_pos[0] + com_vel[0] / self.omega
        xi_y = com_pos[1] + com_vel[1] / self.omega
        
        return np.array([xi_x, xi_y])
    
    def compute_foot_placement(self, com_pos, com_vel, desired_vel):
        """Compute next foot placement"""
        # Current capture point
        xi = self.compute_capture_point(com_pos, com_vel)
        
        # Desired capture point (based on desired velocity)
        T = 0.5  # Step duration
        xi_des = com_pos + desired_vel * T
        
        # Foot should be placed at capture point
        return xi
```

---

## Grasping Fundamentals {#grasping}

### Grasp Quality Metrics

```python
class GraspPlanner:
    def __init__(self):
        self.friction_coefficient = 0.5
    
    def force_closure(self, contact_points, contact_normals):
        """
        Check if grasp achieves force closure
        (can resist arbitrary external forces)
        """
        n_contacts = len(contact_points)
        
        if n_contacts < 4:
            return False  # Need at least 4 contacts in 3D
        
        # Build grasp matrix (simplified)
        G = np.zeros((6, n_contacts * 3))
        
        for i, (point, normal) in enumerate(zip(contact_points, contact_normals)):
            # Force component
            G[:3, i*3:(i+1)*3] = np.eye(3)
            
            # Torque component
            G[3:, i*3:(i+1)*3] = self._skew_symmetric(point)
        
        # Check rank
        rank = np.linalg.matrix_rank(G)
        return rank == 6
    
    def _skew_symmetric(self, v):
        """Skew-symmetric matrix for cross product"""
        return np.array([
            [0,    -v[2],  v[1]],
            [v[2],  0,    -v[0]],
            [-v[1], v[0],  0   ]
        ])
    
    def grasp_quality(self, contact_points, contact_normals):
        """Compute grasp quality metric"""
        # Simplified: minimum distance to grasp wrench space boundary
        wrench_space = self._compute_wrench_space(contact_points, contact_normals)
        
        # Quality = minimum singular value
        _, S, _ = np.linalg.svd(wrench_space)
        quality = np.min(S)
        
        return quality
```

### Grasp Synthesis

```python
class GraspSynthesizer:
    def __init__(self, robot_hand):
        self.hand = robot_hand
    
    def sample_grasps(self, object_mesh, n_samples=100):
        """Sample candidate grasps"""
        grasps = []
        
        for _ in range(n_samples):
            # Sample approach direction
            approach = self._sample_sphere()
            
            # Sample grasp point on object
            grasp_point = self._sample_point_on_mesh(object_mesh)
            
            # Compute grasp pose
            grasp_pose = self._compute_grasp_pose(grasp_point, approach)
            
            # Check collision
            if not self._check_collision(grasp_pose, object_mesh):
                grasps.append(grasp_pose)
        
        return grasps
    
    def rank_grasps(self, grasps, object_mesh):
        """Rank grasps by quality"""
        scores = []
        
        for grasp in grasps:
            # Simulate grasp
            contacts = self._simulate_grasp(grasp, object_mesh)
            
            # Compute quality
            quality = self._grasp_quality(contacts)
            
            scores.append(quality)
        
        # Sort by quality
        ranked_indices = np.argsort(scores)[::-1]
        ranked_grasps = [grasps[i] for i in ranked_indices]
        
        return ranked_grasps, scores
```

---

## Manipulation Planning {#manipulation-planning}

### Pick and Place

```python
from enum import Enum

class ManipulationState(Enum):
    APPROACH = 1
    GRASP = 2
    LIFT = 3
    TRANSPORT = 4
    PLACE = 5
    RETREAT = 6

class PickAndPlaceController:
    def __init__(self, robot):
        self.robot = robot
        self.state = ManipulationState.APPROACH
    
    def execute_pick_and_place(self, object_pose, target_pose):
        """Execute pick and place sequence"""
        
        # 1. Approach
        approach_pose = self._compute_approach_pose(object_pose)
        self.robot.move_to_pose(approach_pose)
        self.state = ManipulationState.GRASP
        
        # 2. Grasp
        self.robot.close_gripper()
        self.state = ManipulationState.LIFT
        
        # 3. Lift
        lift_pose = object_pose.copy()
        lift_pose[2] += 0.1  # 10cm up
        self.robot.move_to_pose(lift_pose)
        self.state = ManipulationState.TRANSPORT
        
        # 4. Transport
        self.robot.move_to_pose(target_pose)
        self.state = ManipulationState.PLACE
        
        # 5. Place
        self.robot.open_gripper()
        self.state = ManipulationState.RETREAT
        
        # 6. Retreat
        retreat_pose = target_pose.copy()
        retreat_pose[2] += 0.1
        self.robot.move_to_pose(retreat_pose)
    
    def _compute_approach_pose(self, object_pose):
        """Compute pre-grasp approach pose"""
        approach = object_pose.copy()
        approach[2] += 0.05  # 5cm above
        return approach
```

### Cartesian Path Planning

```python
class CartesianPlanner:
    def __init__(self, robot):
        self.robot = robot
    
    def plan_straight_line(self, start_pose, end_pose, num_waypoints=50):
        """Plan straight-line Cartesian path"""
        waypoints = []
        
        for alpha in np.linspace(0, 1, num_waypoints):
            # Linear interpolation of position
            pos = (1 - alpha) * start_pose[:3] + alpha * end_pose[:3]
            
            # Spherical linear interpolation of orientation (simplified)
            rot = self._slerp(start_pose[3:], end_pose[3:], alpha)
            
            waypoints.append(np.concatenate([pos, rot]))
        
        return waypoints
    
    def follow_waypoints(self, waypoints, dt=0.1):
        """Execute waypoint trajectory"""
        for waypoint in waypoints:
            # Compute IK
            joint_angles = self.robot.inverse_kinematics(waypoint)
            
            if joint_angles is not None:
                # Move to configuration
                self.robot.set_joint_positions(joint_angles)
                time.sleep(dt)
            else:
                print(f"IK failed for waypoint {waypoint}")
                return False
        
        return True
```

---

## Force Control {#force-control}

### Impedance Control

```python
class ImpedanceController:
    def __init__(self, K_p, K_d):
        """
        Impedance control: F = K_p * (x_d - x) + K_d * (ẋ_d - ẋ)
        K_p: Stiffness matrix (3x3 or 6x6)
        K_d: Damping matrix
        """
        self.K_p = K_p
        self.K_d = K_d
    
    def compute_force(self, x_desired, x_current, v_desired, v_current):
        """Compute desired force"""
        position_error = x_desired - x_current
        velocity_error = v_desired - v_current
        
        force = self.K_p @ position_error + self.K_d @ velocity_error
        
        return force
    
    def compute_joint_torques(self, robot, force_desired):
        """Convert Cartesian force to joint torques"""
        # τ = J^T * F
        J = robot.compute_jacobian(robot.get_joint_positions())
        torques = J.T @ force_desired
        
        return torques
```

### Hybrid Position/Force Control

```python
class HybridController:
    def __init__(self):
        # Selection matrix: 1 = force control, 0 = position control
        self.S_force = np.diag([0, 0, 1, 0, 0, 0])  # Force in Z
        self.S_position = np.eye(6) - self.S_force   # Position in X,Y
    
    def compute_control(self, 
                       x_desired, x_current,
                       f_desired, f_current,
                       K_p, K_f):
        """
        Hybrid position/force control
        K_p: Position gains
        K_f: Force gains
        """
        # Position control
        position_error = x_desired - x_current
        u_position = self.S_position @ K_p @ position_error
        
        # Force control
        force_error = f_desired - f_current
        u_force = self.S_force @ K_f @ force_error
        
        # Combine
        u = u_position + u_force
        
        return u
```

### Compliance Control

```python
class ComplianceController:
    def __init__(self, compliance_matrix):
        """
        Compliance control: x = C * F
        C: Compliance matrix (inverse of stiffness)
        """
        self.C = compliance_matrix
    
    def compute_displacement(self, force_measured):
        """Compute displacement from measured force"""
        displacement = self.C @ force_measured
        return displacement
    
    def admittance_control(self, 
                          x_desired,
                          force_measured,
                          mass_matrix,
                          damping_matrix):
        """
        Admittance control: M*ẍ + D*ẋ + K*x = F_ext
        """
        # Compute acceleration
        x_ddot = np.linalg.inv(mass_matrix) @ (
            force_measured - 
            damping_matrix @ x_dot - 
            np.linalg.inv(self.C) @ (x_current - x_desired)
        )
        
        return x_ddot
```

---

## Integration: Walk and Manipulate {#integration}

### Mobile Manipulation

```python
class MobileManipulator:
    def __init__(self, mobile_base, manipulator):
        self.base = mobile_base
        self.arm = manipulator
    
    def reach_target(self, target_pose):
        """
        Coordinate base and arm to reach target
        """
        # 1. Determine if target is reachable
        reachable, base_pose = self._plan_base_position(target_pose)
        
        if not reachable:
            return False
        
        # 2. Navigate base
        self.base.navigate_to(base_pose)
        
        # 3. Reach with arm
        arm_target = self._transform_to_arm_frame(target_pose, base_pose)
        joint_angles = self.arm.inverse_kinematics(arm_target)
        
        if joint_angles is not None:
            self.arm.move_to_configuration(joint_angles)
            return True
        
        return False
    
    def _plan_base_position(self, target_pose):
        """Plan base position for manipulation"""
        # Keep target within arm reach
        arm_reach = 0.7  # meters
        
        # Compute base position
        base_x = target_pose[0] - arm_reach * np.cos(target_pose[5])
        base_y = target_pose[1] - arm_reach * np.sin(target_pose[5])
        base_theta = target_pose[5]
        
        base_pose = np.array([base_x, base_y, base_theta])
        
        # Check if reachable
        distance = np.linalg.norm(target_pose[:2] - base_pose[:2])
        reachable = distance <= arm_reach
        
        return reachable, base_pose
```

### Whole-Body Task

```python
class WholeBodyManipulation:
    def __init__(self, humanoid_robot):
        self.robot = humanoid_robot
    
    def pick_from_ground(self, object_position):
        """Pick object from ground - requires whole-body motion"""
        
        # 1. Shift weight to one leg
        self._shift_weight('right_leg')
        
        # 2. Bend down
        target_com_height = 0.5  # Lower CoM
        self._lower_com(target_com_height)
        
        # 3. Reach with arm
        joint_angles = self._reach_with_balance(object_position)
        self.robot.set_joint_angles(joint_angles)
        
        # 4. Grasp
        self.robot.close_gripper()
        
        # 5. Stand up
        self._raise_com(0.8)
        
        # 6. Balance weight
        self._shift_weight('center')
    
    def _reach_with_balance(self, target):
        """Compute joint angles that reach target while maintaining balance"""
        # Optimization problem:
        # min ||arm_end - target||²
        # s.t. ZMP inside support polygon
        
        # Use whole-body IK solver
        constraints = {
            'zmp_stability': True,
            'joint_limits': True,
            'self_collision': False
        }
        
        solution = self.robot.whole_body_ik(
            target_pose=target,
            constraints=constraints
        )
        
        return solution
```

---

## Summary

Locomotion and manipulation are the core physical capabilities of humanoid robots. Mastering balance, gait generation, grasping, and force control enables robots to navigate and interact with the world.

**Key Takeaways:**

1. ✅ ZMP-based walking for bipedal stability
2. ✅ Gait generation for smooth locomotion
3. ✅ Force closure for robust grasping
4. ✅ Impedance control for compliant manipulation
5. ✅ Whole-body coordination for complex tasks
6. ✅ Integration of perception, planning, and control

**Next Chapter:**

Chapter 13 brings everything together with Vision-Language-Action models—the future of embodied AI where robots understand language, perceive visually, and act intelligently.

---

*This chapter covered locomotion and manipulation for humanoid robots. You now understand the control principles for walking, balancing, grasping, and coordinated whole-body tasks.*