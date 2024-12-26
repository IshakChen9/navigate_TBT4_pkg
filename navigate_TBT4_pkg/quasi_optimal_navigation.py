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

class QuasiOptimalNavigation(Node):
    def __init__(self):
        super().__init__('lidar_to_base_frame')

        #self.execution_times = []


        timer_period = 0.01  
        self.timer = self.create_timer(timer_period, self.timer_callback)
        # Subscriber to the /scan topic (LIDAR data)
        self.lidar_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )

        # Subscriber to the /odom topic (Odometry data)
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.odometry_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        )

        # Define the publisher for the /cmd_vel topic 
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

        # Initialize the attributes
        self.lidar_ranges = None  # To store the LIDAR ranges
        self.odometry_data = None

        # Define necessary parameters for the LiDAR and obstacle detection
        
        # Initialize the TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Threshold distance between points to consider them part of the same obstacle
        self.obstacle_threshold = 0.1 

        # Angular difference threshold to consider them part of the same obstacle
        self.angular_threshold = math.radians(10)  # 5 degrees in radians

        # Minimum number of points required to consider something an obstacle
        self.min_points_per_obstacle = 5

        # Maximum range for the LIDAR
        self.max_range = 1.5  

        # Robot radius for dilation
        self.robot_radius = 0.3  

        # Define parameters for the control

        # Target's position
        self.target_position = np.array([5, 0.2])  

        # Define the gain of the nominal control Ud
        self.nominal_gain = 1.5  


    def normalize_angle(self, angle):
        """Normalize angle to be between 0 and 2π"""
        if angle < 0:
            angle += 2 * math.pi
        return angle % (2 * math.pi)


    def calculate_angle(self, x, y):
        """Calculate angle in base frame using arctan2"""
        angle = math.atan2(y, x)
        return self.normalize_angle(angle)

    def calculate_dilation(self, distance):
        """Calculate the angular dilation offset using asin(r/d)"""
        if distance <= self.robot_radius:
            return math.pi / 2  
        else:
            return math.asin(self.robot_radius / distance)

    def merge_obstacles_across_zero(self, obstacles):
        """Merge obstacles that wrap around 0 and 360 degrees"""
        if len(obstacles) > 1:
            first_obstacle = obstacles[0]
            last_obstacle = obstacles[-1]

            # Check if one obstacle ends near 360 and another begins near 0
            #if first_obstacle['start_angle'] < 0.1 and last_obstacle['end_angle'] > (2 * math.pi - 0.1):
                # Merge the obstacles
                #merged_obstacle = {
                    #'points': last_obstacle['points'] + first_obstacle['points'],
                    #'start_angle': last_obstacle['start_angle'],
                    #'end_angle': first_obstacle['end_angle']
                #}
                #obstacles = [merged_obstacle] + obstacles[1:-1]
            
            points_s = first_obstacle['points']
            points_e = last_obstacle['points']
            start_angle_s = self.normalize_angle(self.calculate_angle(points_s[0][0], points_s[0][1]))
            end_angle_e = self.normalize_angle(self.calculate_angle(points_e[-1][0], points_e[-1][1]))

            if abs(start_angle_s - end_angle_e) < 0.1:
                # Merge the obstacles
                merged_obstacle = {
                    'points': last_obstacle['points'] + first_obstacle['points'],
                    'start_angle': last_obstacle['start_angle'],
                    'end_angle': first_obstacle['end_angle']
                }
                obstacles = [merged_obstacle] + obstacles[1:-1]

        return obstacles

    def dilate_obstacle(self, obstacle):
        """Dilate the angular range of the obstacle based on the robot's radius"""
        points = obstacle['points']
        start_angle = self.calculate_angle(points[0][0], points[0][1])
        end_angle = self.calculate_angle(points[-1][0], points[-1][1])

        # Distance to the start and end points
        d_s = math.sqrt(points[0][0] ** 2 + points[0][1] ** 2)
        d_e = math.sqrt(points[-1][0] ** 2 + points[-1][1] ** 2)

        # Enlarge the start and end angles
        delta_s = self.calculate_dilation(d_s)
        delta_e = self.calculate_dilation(d_e)
        enlarged_start_angle = self.normalize_angle(start_angle - delta_s)
        enlarged_end_angle = self.normalize_angle(end_angle + delta_e)

        # Check intermediate points to adjust the dilation if needed
        delta_p = abs(enlarged_end_angle - enlarged_start_angle)
        if delta_p > math.pi:
            delta_p = 2 * math.pi - delta_p
        delta_p_0 = delta_p
        index_s = 0  # Index of the point to adjust start angle
        index_e = len(points) - 1  # Index of the point to adjust end angle

        # Adjust by checking other points from the start side
        for i in range(1, len(points) - 2):
            d_i = math.sqrt(points[i][0] ** 2 + points[i][1] ** 2)
            angle_i = self.calculate_angle(points[i][0], points[i][1])
            delta_i = self.calculate_dilation(d_i)
            enlarged_angle_i = self.normalize_angle(angle_i - delta_i)
            delta_n = abs(enlarged_end_angle - enlarged_angle_i)
            if delta_n > math.pi:
                delta_n = 2 * math.pi - delta_n

            if delta_n > delta_p:
                delta_p = delta_n
                index_s = i

        delta_p = delta_p_0

        # Adjust by checking points from the end side
        for j in range(len(points) - 2, 1, -1):
            d_j = math.sqrt(points[j][0] ** 2 + points[j][1] ** 2)
            angle_j = self.calculate_angle(points[j][0], points[j][1])
            delta_j = self.calculate_dilation(d_j)
            enlarged_angle_j = self.normalize_angle(angle_j + delta_j)
            delta_n = abs(enlarged_angle_j - enlarged_start_angle)
            if delta_n > math.pi:
                delta_n = 2 * math.pi - delta_n

            if delta_n > delta_p:
                delta_p = delta_n
                index_e = j

        # Compute the dilated angles
        start_point = points[index_s]
        end_point = points[index_e]
        d_index_s = math.sqrt(start_point[0] ** 2 + start_point[1] ** 2)
        d_index_e = math.sqrt(end_point[0] ** 2 + end_point[1] ** 2)

        # Ensure dilation when distances are <= robot radius
        final_start_angle = self.normalize_angle(self.normalize_angle(self.calculate_angle(start_point[0], start_point[1])) - self.calculate_dilation(d_index_s))
        final_end_angle = self.normalize_angle(self.normalize_angle(self.calculate_angle(end_point[0], end_point[1])) + self.calculate_dilation(d_index_e))

        # Return the dilated obstacle
        return {
            'points': points,
            'start_angle': final_start_angle,
            'end_angle': final_end_angle
        }
    

    def active_portion(self):
        """Determine the detected obstacle segment to be avoided"""
        obstacles = self.dilated_obstacles
        if len(obstacles) != 0:
            Xd = self.target_position  
            X = self.X  
            vd = [Xd[0] - X[0], Xd[1] - X[1]]  
            yaw = self.normalize_angle(self.yaw)  
            phi_d = self.calculate_angle(vd[0], vd[1])
            delta_phi = self.normalize_angle(phi_d - yaw) 
            indices = []

            # Loop through each obstacle and check if it's in the path
            for i in range(len(obstacles)):
                start_angle = obstacles[i]['start_angle']
                end_angle = obstacles[i]['end_angle']

                # If start_angle < end_angle, we're dealing with a normal arc
                if start_angle < end_angle:
                    if (start_angle <= delta_phi <= end_angle):
                        indices.append(i)
                # Handle obstacle arcs that wrap around 0 degrees (i.e., from large angle to small angle)
                else:
                    if delta_phi >= start_angle or delta_phi <= end_angle:
                        indices.append(i)

            # Determine the active obstacle if any obstacles block the path
            if len(indices) < 1:
                index = None 
            elif len(indices) == 1:
                index = indices[0]  # Only one obstacle blocking the path
            else:
                # If multiple obstacles, find the closest one based on minimum distance
                min_arcs = []
                for i in indices:
                    points = obstacles[i]['points']
                    min_distance = min(math.sqrt(p[0] ** 2 + p[1] ** 2) for p in points)
                    min_arcs.append(min_distance)

                # Find the index of the obstacle with the minimum distance
                index = indices[min_arcs.index(min(min_arcs))]

        else:
            index = None  

        return index
    
    def virtual_center(self, active_obstacle_index):
        """Determine the virtual center, virtual angle for the control"""
        active_obstacle = self.dilated_obstacles[active_obstacle_index]
        obstacle_points = active_obstacle['points']
        min_distance = float('inf')
        closest_point_index = None

        # Step 1: Find the closest point in the robot frame (base link frame)
        for i, point in enumerate(obstacle_points):
            point_x, point_y = point 
            distance = math.sqrt(point_x ** 2 + point_y ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_point_index = i

        # Step 2: Transform the closest point from the robot frame to the global frame
        if closest_point_index is not None:
            closest_point_x, closest_point_y = obstacle_points[closest_point_index]
            angle = self.calculate_angle(closest_point_x, closest_point_y)
            X = self.X
            robot_x = X[0] 
            robot_y = X[1] 
            robot_yaw = self.yaw  
            global_x = robot_x + (closest_point_x * math.cos(robot_yaw) - closest_point_y * math.sin(robot_yaw))
            global_y = robot_y + (closest_point_x * math.sin(robot_yaw) + closest_point_y * math.cos(robot_yaw))

            return (global_x, global_y), min_distance, angle
        else:
            return None, None, None



    def odometry_callback(self, msg: Odometry):
        #print("Odom")
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
            obstacles = []
            current_obstacle = []

            # Previous point for distance and angular comparison
            prev_point = None
            prev_angle = None

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

                    # Calculate the angle of the current point in the base frame
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

                        # Compute the angular difference with the previous point
                        #angular_difference = abs(current_angle - prev_angle)
                        angular_difference = abs(angle - prev_angle)

                        # Check if the current point belongs to the same obstacle
                        if distance <= self.obstacle_threshold or  (distance > self.obstacle_threshold and angular_difference <= self.angular_threshold):
                            # Add point to the current obstacle
                            current_obstacle.append((point_in_base.point.x, point_in_base.point.y))
                            prev_point = point_in_base
                            #prev_angle = current_angle
                            prev_angle = angle
                        else:
                            # Process the obstacle: calculate start and end angles using arctan2 in base frame
                            if len(current_obstacle) >= self.min_points_per_obstacle:
                                # Dilate the obstacle using the method
                                dilated_obstacle = self.dilate_obstacle({
                                    'points': current_obstacle
                                })
                                obstacles.append(dilated_obstacle)

                            # Start a new obstacle
                            current_obstacle = [(point_in_base.point.x, point_in_base.point.y)]
                            prev_point = point_in_base
                            #prev_angle = current_angle
                            prev_angle = angle

            # Handle the last obstacle
            if current_obstacle and len(current_obstacle) >= self.min_points_per_obstacle:
                dilated_obstacle = self.dilate_obstacle({
                    'points': current_obstacle
                })
                obstacles.append(dilated_obstacle)

            # Sort obstacles by start_angle
            obstacles = sorted(obstacles, key=lambda x: x['start_angle'])

            # Merge obstacles that wrap around 0 and 360 degrees
            obstacles = self.merge_obstacles_across_zero(obstacles)

            self.dilated_obstacles = obstacles

            #self.get_logger().info(f"Dilated Obstacles: {obstacles}")

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
            with open('execution_time_stats_QOpt.csv', mode='w', newline='') as file:
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
        #tester = False
        #tester_2= True
        #start_time = time.time()
        if self.lidar_ranges is not None and self.odometry_data is not None:
            #tester = True
            Xd = self.target_position
            X = self.X
            ud = self.nominal_gain * (Xd - X)
            yaw = self.normalize_angle(self.yaw)
            phi_d = self.calculate_angle(ud[0],ud[1])
            delta_phi = self.normalize_angle(phi_d - yaw)

            if LA.norm(X - Xd) < 0.02:
                #tester_2 = False
                linear_velocity = 0.0
                angular_velocity = 0.0
                #self.save_execution_time_statistics()


            else:
                index = self.active_portion()
                dilated_obstacles = self.dilated_obstacles
                if index is not None:
                    v_center, v_center_distance, virtual_center_angle = self.virtual_center(index)
                    vcd = (v_center - Xd)
                    vxd = -ud
                    if np.dot(vcd, vxd) >= 0:
                        Vc = (v_center - X) / v_center_distance
                        if dilated_obstacles[index]['start_angle'] < dilated_obstacles[index]['end_angle']:
                            if delta_phi > virtual_center_angle:
                                theta = dilated_obstacles[index]['end_angle'] - virtual_center_angle
                            else:
                                theta = virtual_center_angle - dilated_obstacles[index]['start_angle']
                            beta = abs(delta_phi - virtual_center_angle)
                        else:
                            th1 = 2 * math.pi - dilated_obstacles[index]['start_angle'] + dilated_obstacles[index]['end_angle']
                            if virtual_center_angle > dilated_obstacles[index]['start_angle']:
                                thc = virtual_center_angle - dilated_obstacles[index]['start_angle']
                            else:
                                thc = 2 * math.pi - dilated_obstacles[index]['start_angle'] + virtual_center_angle
                            if delta_phi > dilated_obstacles[index]['start_angle']:
                                thd = delta_phi - dilated_obstacles[index]['start_angle']
                            else:
                                thd = 2 * math.pi - dilated_obstacles[index]['start_angle'] + delta_phi
                            if thd > thc:
                                theta = th1 - thc
                            else:
                                theta = thc
                            beta = abs(thd - thc)
                        u = ud - LA.norm(ud) * math.sin(theta - beta) / math.sin(theta) * Vc
                    else:
                        u = ud
                else:
                    u = ud
                theta_d = math.atan2(u[1], u[0])
                angle_diff = -theta_d + self.yaw
                if angle_diff <= (-math.pi):
                        angle_diff += 2 * math.pi
                if angle_diff > math.pi:
                        angle_diff -= 2 * math.pi
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
            #file_path = "robot_trajectory_quasi_opt.csv"

            #if not os.path.isfile(file_path):
               # with open(file_path, mode='w', newline='') as file:
                    #writer = csv.writer(file)
                    #writer.writerow(["X", "Y"])  # Write header if file is new

            # Append the robot's current position to the CSV file
            #with open(file_path, mode='a', newline='') as file:
                #writer = csv.writer(file)
                #writer.writerow([X[0], X[1]])  # Write (x, y) data


def main(args=None):
    rclpy.init(args=args)
    node = QuasiOptimalNavigation()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
