import math
#import time
#import statistics
from numpy import linalg as LA
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Twist
from tf_transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
from rclpy.qos import ReliabilityPolicy, QoSProfile
#import csv
#import os


class HybridNavigation(Node):
    def __init__(self):
        super().__init__('lidar_to_base_frame')

        #self.execution_times = []

        # Create timer subscriber
        timer_period = 0.01  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Subscriber to the /scan topic 
        self.lidar_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )

        # Subscriber to the /odom topic 
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.odometry_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        )

        # Initialize the attributes
        self.lidar_ranges = None  
        self.odometry_data = None 

        # Define the publisher for the /cmd_vel topic 
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Define necessary parameters for the LiDAR and obstacle detection

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Threshold distance between points to consider them part of the same obstacle
        self.obstacle_threshold = 0.1  

        # Minimum number of points required to consider something an obstacle
        self.min_points_per_obstacle = 5

        # Maximum range for the LIDAR
        self.max_range = 1.5  

        # Robot radius for dilation
        self.robot_radius = 0.3 

        #Safety margin for obstacles reconstruction
        self.e_safety = 0.05

        # Define parameters for the control

        # Design parameter for the virtual destinations
        self.e_parameter = 0.6

        # Smoothness parameter
        self.epsilon = 0.05

        # Target's position
        self.target_position = np.array([5, 0.2])  

        # Define the gain of the nominal control Ud
        self.nominal_gain = 1.5 

        # Define the variable to store the active obstacle
        self.active_center_t = None
        self.active_radius_t = None
        self.active_dilated_radius_t = None
        self.obstacle_range_t = None



    def normalize_angle(self, angle):
        """Normalize angle to be between 0 and 2π"""
        if angle < 0:
            angle += 2 * math.pi
        return angle % (2 * math.pi)

    def calculate_angle(self, x, y):
        """Calculate angle in base frame using arctan2"""
        angle = math.atan2(y, x)
        return self.normalize_angle(angle)

    def merge_obstacles_across_zero(self, obstacles):
        """Merge obstacles that wrap around 0 and 360 degrees"""
        if len(obstacles) > 1:
            first_obstacle = obstacles[0]
            last_obstacle = obstacles[-1]            
            points_s = first_obstacle['points']
            points_e = last_obstacle['points']
            start_angle_s = self.normalize_angle(self.calculate_angle(points_s[0][0], points_s[0][1]))
            end_angle_e = self.normalize_angle(self.calculate_angle(points_e[-1][0], points_e[-1][1]))

            if abs(start_angle_s - end_angle_e) < 0.1:
                merged_obstacle = {
                    'points': last_obstacle['points'] + first_obstacle['points'],
                    'start_angle': last_obstacle['start_angle'],
                    'end_angle': first_obstacle['end_angle']
                }
                obstacles = [merged_obstacle] + obstacles[1:-1]

        return obstacles

    def obstacles_reconstruction(self):
        """ Reconstruct obstacles from the detected arcs"""
        detected_obstacles = self.detected_obstacles
        if len(detected_obstacles) != 0:
            obstacles_centers = []
            obstacles_radii = []
            for i in range(len(detected_obstacles)):
                points = detected_obstacles[i]['points']
                starting_point = np.array([detected_obstacles[i]['points'][0][0], detected_obstacles[i]['points'][0][1]])
                ending_point = np.array([detected_obstacles[i]['points'][-1][0], detected_obstacles[i]['points'][-1][1]])
                # Calculate the minimum distance (norm) and find the index of the closest point
                min_distance, closest_point_index = min(
                    (math.sqrt(p[0] ** 2 + p[1] ** 2), index) for index, p in enumerate(points)
                )
                
                # Closest point in base_link (robot) frame
                closest_point_x, closest_point_y = points[closest_point_index]
                closest_pt = np.array([closest_point_x, closest_point_y])
                dist_s = LA.norm(closest_pt - starting_point)
                dist_e = LA.norm(closest_pt - ending_point)
                # Determine the obstacle radius and center
                half_distance_start_end_points = LA.norm(starting_point - ending_point) / 2
                average_distance_to_closest_point_end_start = (dist_s + dist_e) / 2
                distance_closestpt_to_line_start_end = math.sqrt(average_distance_to_closest_point_end_start ** 2 - half_distance_start_end_points ** 2)
                if distance_closestpt_to_line_start_end > 0.02:
                    obstacle_radius = average_distance_to_closest_point_end_start ** 2 / (2 * distance_closestpt_to_line_start_end) 
                    obstacle_center_robot_frame = [closest_point_x * (1 + obstacle_radius / min_distance) , closest_point_y * (1 + obstacle_radius / min_distance)]

                    # Get robot's position and orientation (yaw) in the global frame
                    X = self.X
                    robot_yaw = self.yaw 

                    # Apply the transformation to the global frame 
                    global_x = X[0] + (obstacle_center_robot_frame[0] * math.cos(robot_yaw) - obstacle_center_robot_frame[1] * math.sin(robot_yaw))
                    global_y = X[1] + (obstacle_center_robot_frame[0] * math.sin(robot_yaw) + obstacle_center_robot_frame[1] * math.cos(robot_yaw))

                    obstacles_radii.append(obstacle_radius + self.e_safety)
                    obstacles_centers.append([global_x, global_y])
        else:
            obstacles_radii = None
            obstacles_centers = None
        
        return obstacles_centers, obstacles_radii


    def active_obstacle(self):
        """ Determine the obstacle that must be avoided """
        obstacles_centers, obstacles_radii = self.obstacles_reconstruction()
        if obstacles_radii is not None:
            Xd = self.target_position  
            X = self.X  
            indices = []
            angles_phi_d = []
            # Loop through each obstacle and check if it's in the path
            for i in range(len(obstacles_radii)):
                phi_d = math.asin((obstacles_radii[i] + self.robot_radius) / LA.norm(obstacles_centers[i] - Xd))
                t = np.dot(obstacles_centers[i] - Xd, X - Xd) / (LA.norm(obstacles_centers[i] - Xd) * LA.norm(X - Xd))
                z = np.dot(Xd - X, obstacles_centers[i] - X)
                # cheking if the robot is inside the active region of obstacle of index i
                if (t >= math.cos(phi_d)) and (z >= 0):
                    indices.append(i)
                angles_phi_d.append(phi_d)

            # Determine the active obstacle if any obstacles block the path
            if len(indices) < 1:
                obstacle = None  
                radius = None
                obstacle_range = None
            else:
                min_distance, closest_obstacle_index = min(
                    (math.sqrt((X[0] - obstacles_centers[i][0]) ** 2 + (X[1] - obstacles_centers[i][1]) ** 2) - obstacles_radii[i], i)
                    for i in indices 
                )
                obstacle = obstacles_centers[closest_obstacle_index]
                radius = obstacles_radii[closest_obstacle_index]
                index_shadowed_obstacles = []
                for i in range(len(obstacles_radii)):
                    theta_i = math.cos(np.dot(obstacle - Xd, obstacles_centers[i] - Xd) / (LA.norm(obstacle - Xd) * LA.norm(obstacles_centers[i] - Xd)))
                    phi_c = theta_i - angles_phi_d[i]
                    if (phi_c <= angles_phi_d[closest_obstacle_index]) and (i != closest_obstacle_index):
                        index_shadowed_obstacles.append(i)
                
                if len(index_shadowed_obstacles) >= 1:
                    obstacle_range = min(
                      math.sqrt((obstacle[0] - obstacles_centers[i][0]) ** 2 + (obstacle[1] - obstacles_centers[i][1]) ** 2) - obstacles_radii[i] - radius - 2 * self.robot_radius
                      for i in index_shadowed_obstacles
                    )
                else:
                    obstacle_range = None

        else:
            obstacle = None  
            radius = None
            obstacle_range = None
        if obstacle_range is None:
            obstacle_range = self.max_range - self.e_safety - self.robot_radius
        return obstacle, radius, obstacle_range
    
    def fully_actuated_control(self):
        """ Calculate the velocity control for the 1st order fully actuated model"""
        X = self.X
        Xd = self.target_position
        ud = self.nominal_gain * (Xd - X)
        active_center, active_radius, obstacle_range = self.active_obstacle()
        if (self.active_center_t is not None) and (active_center is not None):
            if LA.norm(np.array(active_center) - np.array(self.active_center_t)) > self.active_radius_t:
                self.active_center_t = active_center
                self.active_radius_t = active_radius
                self.obstacle_range_t = obstacle_range
        else:
            self.active_center_t = active_center
            self.active_radius_t = active_radius
            self.obstacle_range_t = obstacle_range
        
        if (active_center is not None) and (LA.norm(active_center - X) - active_radius - self.robot_radius <= self.obstacle_range_t):
            phi = math.asin((self.active_radius_t + self.robot_radius) / LA.norm(self.active_center_t - Xd))
            rot_matrix1 = [[math.cos(phi), -math.sin(phi)] , [math.sin(phi), math.cos(phi)]]
            rot_matrix2 = [[math.cos(-phi), -math.sin(-phi)] , [math.sin(-phi), math.cos(-phi)]]
            v1 = (self.active_center_t - Xd) / LA.norm(self.active_center_t - Xd)
            xd1 = Xd + self.e_parameter * np.matmul(rot_matrix1, v1)
            xd2 = Xd + self.e_parameter * np.matmul(rot_matrix2, v1)
            distance_ratio = (self.active_radius_t + self.robot_radius) / LA.norm(self.active_center_t - X)
            clamped_ratio = min(1, distance_ratio)  
            theta = math.asin(clamped_ratio)
            Vd1 = (xd1 - X) / LA.norm(xd1 - X)
            Vd2 = (xd2 - X) / LA.norm(xd2 - X)
            Vc = (self.active_center_t - X) / LA.norm(self.active_center_t - X)
            beta1 = math.acos(np.dot(Vd1, Vc))
            beta2 = math.acos(np.dot(Vd2, Vc))
            rot_matrix = [[0, 1] , [-1, 0]]
            v = np.matmul(rot_matrix, Xd - self.active_center_t)
            d = (self.obstacle_range_t - LA.norm(self.active_center_t - X) + self.active_radius_t + self.robot_radius) / self.epsilon
            if (np.dot(Vc, v) <= 0):
                u = self.nominal_gain * min(1, d) * (LA.norm(X - xd1) + (beta1 / theta) * LA.norm(Xd - xd1)) * (Vd1 + (- math.cos(beta1) + math.cos(theta) * math.sin(beta1) / math.sin(theta)) * Vc) + (1 - min(1, d)) * ud
            else:
                u = self.nominal_gain * min(1, d) * (LA.norm(X - xd2) + (beta2 / theta) * LA.norm(Xd - xd2)) * (Vd2 + (- math.cos(beta2) + math.cos(theta) * math.sin(beta2) / math.sin(theta)) * Vc) + (1 - min(1, d)) * ud
        else:
            u = ud
        return u


    def odometry_callback(self, msg: Odometry):
        self.odometry_data = msg.pose.pose
        self.X = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, self.yaw = euler_from_quaternion(orientation_list)
    

    def lidar_callback(self, msg: LaserScan):
        """ Recieve and process the Lidar data """
        try:
            # Get the transformation from 'base_link' to 'rplidar_link'
            transform = self.tf_buffer.lookup_transform(
                'base_link',  
                'rplidar_link', 
                rclpy.time.Time() 
            )

            self.lidar_ranges = msg.ranges

            # List to hold all obstacles
            obstacles = []

            # Temporary list to hold points for the current obstacle
            current_obstacle = []

            # Previous point for distance and angular comparison
            prev_point = None
            #prev_angle = None

            for i, range_value in enumerate(msg.ranges):
                if range_value >= msg.range_min and range_value <= min(msg.range_max, self.max_range):
                    # Calculate the angle of the point in radians (from lidar frame)
                    angle = msg.angle_min + i * msg.angle_increment

                    # Create a PointStamped in the lidar frame
                    point_in_lidar = PointStamped()
                    point_in_lidar.header.frame_id = 'rplidar_link'
                    point_in_lidar.point.x = range_value * math.cos(angle)
                    point_in_lidar.point.y = range_value * math.sin(angle)
                    point_in_lidar.point.z = 0.0

                    # Transform the point to the base frame
                    point_in_base = tf2_geometry_msgs.do_transform_point(point_in_lidar, transform)

                    #current_angle = self.calculate_angle(point_in_base.point.x, point_in_base.point.y)

                    # If this is the first point of a potential obstacle
                    if not current_obstacle:
                        current_obstacle.append((point_in_base.point.x, point_in_base.point.y))
                        prev_point = point_in_base
                        #prev_angle = current_angle
                        prev_angle = angle
                    else:
                        # Compute the distance to the previous point
                        distance = math.sqrt(
                            (point_in_base.point.x - prev_point.point.x) ** 2 +
                            (point_in_base.point.y - prev_point.point.y) ** 2
                        )

                        #angular_difference = abs(angle - prev_angle) 

                        # Check if the current point belongs to the same obstacle
                        if distance <= self.obstacle_threshold: #or (distance >= self.obstacle_threshold and angular_difference <= angular_threshold)
                            current_obstacle.append((point_in_base.point.x, point_in_base.point.y))
                            prev_point = point_in_base
                            #prev_angle = current_angle
                            #prev_angle = angle
                        else:
                            # Process the obstacle: calculate start and end angles using arctan2 in base frame
                            if len(current_obstacle) >= self.min_points_per_obstacle:
                                start_angle = self.calculate_angle(current_obstacle[0][0], current_obstacle[0][1])
                                end_angle = self.calculate_angle(current_obstacle[-1][0], current_obstacle[-1][1])

                                obstacles.append({
                                    'points': current_obstacle,
                                    'start_angle': start_angle,
                                    'end_angle': end_angle
                                })

                            # Start a new obstacle
                            current_obstacle = [(point_in_base.point.x, point_in_base.point.y)]
                            prev_point = point_in_base
                            prev_angle = angle

            # Handle the last obstacle
            if current_obstacle and len(current_obstacle) >= self.min_points_per_obstacle:
                start_angle = self.calculate_angle(current_obstacle[0][0], current_obstacle[0][1])
                end_angle = self.calculate_angle(current_obstacle[-1][0], current_obstacle[-1][1])
                obstacles.append({
                    'points': current_obstacle,
                    'start_angle': start_angle,
                    'end_angle': end_angle
                })

            # Sort obstacles by start_angle
            obstacles = sorted(obstacles, key=lambda x: x['start_angle'])

            # Merge obstacles that wrap around 0 and 360 degrees
            obstacles = self.merge_obstacles_across_zero(obstacles)
            self.detected_obstacles = obstacles

            #self.get_logger().info(f"Detected Obstacles: {obstacles}")

        except Exception as e:
            self.get_logger().error(f"Failed to transform lidar points: {e}")

    """    
    def save_execution_time_statistics(self):
        #Compute and save statistics for execution times.
        mean_time = statistics.mean(self.execution_times)
        stdev_time = statistics.stdev(self.execution_times)

        # Log statistics
        #self.get_logger().info(f"Average Execution Time: {mean_time:.6f} seconds")
        #self.get_logger().info(f"Standard Deviation: {stdev_time:.6f} seconds")

        # Save to file
        with open('execution_time_stats_Hyb.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Sample Index', 'Execution Time (s)'])
            for idx, exec_time in enumerate(self.execution_times):
                writer.writerow([idx + 1, exec_time])
            writer.writerow([])
            writer.writerow(['Average Time', 'Standard Deviation'])
            writer.writerow([mean_time, stdev_time])

        #self.get_logger().info("Saved execution time statistics to 'execution_time_stats.csv'.")
    """

    def timer_callback(self):
        """ Transform the fully actuated control into differential drive control and publish"""
        #tester = False
        #tester_2 = True
        #start_time = time.time()
        if self.lidar_ranges is not None and self.odometry_data is not None:
            #tester = True
            Xd = self.target_position
            X = self.X
            if LA.norm(Xd - X) < 0.02:
                #tester_2 = False
                linear_velocity = 0.0
                angular_velocity = 0.0
                #self.save_execution_time_statistics()

            else:
                u = self.fully_actuated_control()
                theta_d = math.atan2(u[1], u[0])
                angle_diff = -theta_d + self.yaw
                if angle_diff <= (-math.pi):
                        angle_diff = angle_diff + 2 * math.pi
                if angle_diff > math.pi:
                        angle_diff = -2 * math.pi + angle_diff
                linear_velocity = min(0.31, 0.1 * LA.norm(u) * (1 + math.cos(angle_diff)) / 2)
                ang_vel =  -0.8 * math.sin(angle_diff) - 0.8 * angle_diff
                ang_vel_abs = min(abs(ang_vel), 1.9)
                angular_velocity = np.sign(ang_vel) * ang_vel_abs

        else:
            linear_velocity = 0.0
            angular_velocity = 0.0
        msg = Twist()
        msg.linear.x = linear_velocity
        msg.angular.z = angular_velocity
        self.publisher_.publish(msg)

        #end_time = time.time()
        #loop_time = end_time - start_time
        #if tester and tester_2:
            #self.execution_times.append(loop_time)

            #file_path = "robot_trajectory_hyb.csv"

            #if not os.path.isfile(file_path):
                #with open(file_path, mode='w', newline='') as file:
                    #writer = csv.writer(file)
                    #writer.writerow(["X", "Y"])  

            #with open(file_path, mode='a', newline='') as file:
                #writer = csv.writer(file)
                #writer.writerow([X[0], X[1]]) 



def main(args=None):
    rclpy.init(args=args)
    node = HybridNavigation()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
