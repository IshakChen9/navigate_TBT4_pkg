# navigate_TBT4_pkg

## Introduction
This is a ROS2 package containing four navigation algorithms that were developped for a Turtlebot4. The four navigation algorithms are presented as follows:
1. The hybrid feedback navigation approach (local_optimal_hybrid_navigation.py)

   This approach is applicable in environments with disk-shaped obstacles.
   
   - “Global Hybrid Feedback Control with Local Optimal Obstacle Avoidance Maneuvers.”
   
2. The quai-optimal navigation approach (quasi_optimal_navigation.py)

   This approach applies to environments with convex obstacles satisfying a curvature condition (see the reference below for more details).
   
   - Cheniouni, I., Berkane, S., and Tayebi, A., “Safe and Quasi-Optimal Autonomous Navigation in Environments With Convex Obstacles,” IEEE Transactions on Automatic Control, 2024.
   
2. The separating hyperplane navigation approach (separating_hyperplane_approach.py)

   This approach applies to environments with convex obstacles satisfying a curvature condition (see the reference below for more details).
   
   - Arslan, O., and Koditschek, D. E., “Sensor-based reactive navigation in unknown convex sphere worlds,” The International Journal of Robotics Research, vol. 38, no. 2-3, pp. 196–223, 2019.
   
3. The vector field histogram approach (VFH_navigation.py)

   This approach applies to environments with arbitrary shapes but requires careful tuning of a parameter (self.threshold) that depends on the workspace (see the reference below for more details). In our experiments, 70000 was a decent value.
   
   - Borenstein, J., and Koren, Y., “The vector field histogram-fast obstacle avoidance for mobile robots,” IEEE Transactions on Robotics and Automation, vol. 7, no. 3, pp. 278–288, 1991  

## Prerequisites
1. Turtlebot4 (or other version with the appropriate modification)
2. Ubuntu 22.04/ROS2_Humble (or other distros with the appropriate modifications)
3. User PC

## Getting started
1. Setup your robot following the user manual https://turtlebot.github.io/turtlebot4-user-manual/#turtlebot4-user-manual
2. Clone the package into your workspace
3. Navigate to your workspace:
   ####
       cd ~/ros2_ws
   
5. Build the package:
   ####
       colcon build --symlink-install --packages-select navigate_TBT4_pkg
   
7. Source the workspace:
   ####
       source install/setup.bash
   
8. Run one of the desired navigation codes:
   ####
       ros2 run navigate_TBT4_pkg local_optimal_hybrid_navigation

## Remarks
1. In our experiments, we used the simple discovery (networking) mode (see the user manual in the previous subsection for more details).
2. The different tuning and control design parameters must be adjusted according to the workspace and needs of the user.
3. The results of some experiments are reported in our journal paper “Global Hybrid Feedback Control with Local Optimal Obstacle Avoidance Maneuvers.” They can be visualised in the following videos:
   - The hybrid feedback navigation approach: https://youtu.be/rQc062EDYts
   - The quai-optimal navigation approach: https://youtu.be/Z2AWva6DYgs
   - A comparative study of the four approaches: https://youtu.be/KzUNLwQ5lMo
   



