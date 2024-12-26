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


class VFHNavigation(Node):
    def __init__(self):
        super().__init__('lidar_to_base_frame')

        #self.execution_times = []

        timer_period = 0.01 
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.lidar_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )

        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.odometry_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        )


        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

        # Initialize the attributes
        self.lidar_ranges = None  
        self.odometry_data = None 
        self.lidar_data_base_frame = None

        # Define necessary parameters for the LiDAR and obstacle detection
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.max_range = 1.5


        # Define parameters for the grid and control
        self.robot_radius = 0.3  
        
        self.grid_size = 200
        self.window_size = 33
        self.cartesian_grid = np.zeros((self.grid_size, self.grid_size))
        self.cell_size = 0.10
        self.alpha = 5
        self.grid_robot_position = [0, 0]
        self.threshold = 70000
        self.min_size = 5
        self.max_size = 18

        self.target_position = np.array([6.6, -3]) 

        self.steering_gain = 0.8
        self.h_m = 1000
        self.v_max = 0.31
        self.a_max = 1.9
        self.v_min = 0.04
        self.l = 3
        self.v_gain = 0.5



    def normalize_angle(self, angle):
        """Normalize angle to be between 0 and 2π"""
        if angle < 0:
            angle += 2 * math.pi
        return angle % (2 * math.pi)
    
    def normalize_angle2(seldf, angle):
        """Normalize angle to be between -pi and π"""
        if angle > math.pi:
            angle -= 2 * math.pi
        elif angle <= -math.pi:
            angle += 2 * math.pi
        return angle

    def update_cartesian_histogram(self, points):
        half_grid = self.grid_size // 2

        if len(points) > 0:
            for x, y in points:
                grid_x = int(x / self.cell_size) + half_grid
                grid_y = int(y / self.cell_size) + half_grid
                if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                    self.cartesian_grid[grid_y, grid_x] += 1 
    
    def extract_active_region(self):
        robot_x, robot_y = self.X
        half_ws = self.window_size // 2  
        grid_half = self.grid_size // 2

        grid_robot_x = int(robot_x / self.cell_size) + grid_half
        grid_robot_y = int(robot_y / self.cell_size) + grid_half
        self.grid_robot_position = [grid_robot_x, grid_robot_y]

        start_x = max(0, grid_robot_x - half_ws)
        end_x = min(self.grid_size, grid_robot_x + half_ws)
        start_y = max(0, grid_robot_y - half_ws)
        end_y = min(self.grid_size, grid_robot_y + half_ws)

        return self.cartesian_grid[start_y:end_y, start_x:end_x]
    
    def update_polar_histogram(self):
        active_region = self.extract_active_region()
        x_0, y_0 = self.grid_robot_position 
        num_sectors = int(360 / self.alpha)
        polar_histogram = np.zeros(num_sectors)

        d_max = math.sqrt(2) * (self.window_size - 1) / 2  
        a = d_max / 2
        b = 0.5
        for i in range(active_region.shape[0]):
            for j in range(active_region.shape[1]):
                c_star = active_region[i, j]
                if c_star > 0:  
                    x_i = (j - self.window_size // 2) * self.cell_size + x_0
                    y_i = (i - self.window_size // 2) * self.cell_size + y_0

                    d_ij = math.sqrt((x_i - x_0) ** 2 + (y_i - y_0) ** 2)
                    beta_ij = math.degrees(self.normalize_angle(math.atan2(y_i - y_0, x_i - x_0)))

                    k = int(beta_ij / self.alpha) 

                    m_ij = (c_star ** 2) * (a - b * d_ij) 

                    polar_histogram[k] += m_ij

        smoothed_histogram = np.zeros(num_sectors)
        for k in range(num_sectors):
            for offset in range(-self.l, self.l):
                if k + offset < 0:
                    neighbor = num_sectors + k + offset
                elif k + offset >= num_sectors:
                    neighbor = k + offset - num_sectors
                else:
                    neighbor = k + offset
                smoothed_histogram[k] += (self.l - abs(offset) + 1) * polar_histogram[neighbor]
            smoothed_histogram[k] /= (2 * self.l + 1)

        return smoothed_histogram

    def candidate_valleys(self, polar_histogram):
        valleys = []
        in_valley = False
        valley_start = None
        num_sectors = int(360 / self.alpha)
        # Traverse the histogram to find valleys
        for i in range(num_sectors):
            if polar_histogram[i] < self.threshold:
                if not in_valley:
                    # Start a new valley
                    in_valley = True
                    valley_start = i
            else:
                if in_valley:
                    # End the current valley
                    in_valley = False
                    valleys.append((valley_start, i - 1))

        # If the histogram ends while still in a valley, close the valley
        if in_valley:
            valleys.append((valley_start, num_sectors - 1))

        # Handle wrap-around valleys
        if len(valleys) > 1:
            first_valley = valleys[0]
            last_valley = valleys[-1]

            # Check if the last valley ends at the boundary and the first starts at 0
            if first_valley[0] == 0 and last_valley[1] == num_sectors - 1:
                # Merge the two valleys into one
                merged_valley = (last_valley[0], first_valley[1])
                valleys = valleys[1:-1]  # Remove the first and last valleys
                valleys.insert(0, merged_valley)  # Insert the merged valley at the start
        
        return valleys


    def selected_valley(self, valleys):
        if len(valleys) < 1:
            return None, None, None
        num_sectors = int(360 / self.alpha)
        v_d = self.target_position - self.X
        beta_d = math.degrees(self.normalize_angle(math.atan2(v_d[1], v_d[0])))
        target_sector = int(beta_d / self.alpha)
        inside_valley = False
        closest_valley = None
        closest_sector_index = None

        min_distance = float('inf')

        for start, end in valleys:
            distance_to_start = min(abs(target_sector - start), num_sectors - abs(target_sector - start))
            distance_to_end = min(abs(target_sector - end), num_sectors - abs(target_sector - end))
            if start <= end:
                if start <= target_sector <= end:
                    inside_valley = True
                    closest_valley = (start, end)
                    closest_sector_index = 0 if distance_to_start < distance_to_end else 1
                    return closest_valley, closest_sector_index, inside_valley
            else:
                if target_sector >= start or target_sector <= end:
                    inside_valley = True
                    closest_valley = (start, end)
                    closest_sector_index = 0 if distance_to_start < distance_to_end else 1
                    return closest_valley, closest_sector_index, inside_valley

            distance = min(distance_to_start, distance_to_end)

            if distance < min_distance:
                min_distance = distance
                closest_valley = (start, end)
                closest_sector_index = 0 if distance_to_start == distance else 1
            
        return closest_valley, closest_sector_index, inside_valley

    def steering_angle(self, closest_sector_index, valley, inside_valley):
        if valley is None:
            return None
        v_d = self.target_position - self.X
        beta_r = self.normalize_angle(math.atan2(v_d[1], v_d[0]))
        beta_d = math.degrees(beta_r)
        target_sector = int(beta_d / self.alpha)
        num_sectors = int(360 / self.alpha)
        start, end = valley
        
        valley_size = end - start + 1 if end >= start else end + num_sectors - start + 1

        if valley_size >= num_sectors:
            return beta_r
        if inside_valley:
            if end >= start:
                if min(target_sector - start, end - target_sector) >=  self.min_size:
                    return beta_r
                elif (target_sector - start) < self.min_size and valley_size >= 2 * self.min_size:
                    return self.normalize_angle(math.radians((start + self.min_size) * self.alpha))
                elif (end - target_sector) < self.min_size and valley_size >= 2 * self.min_size:
                    return self.normalize_angle(math.radians((end - self.min_size) * self.alpha))
                else:
                    if valley_size > self.max_size:
                        k_f = valley[closest_sector_index] - self.max_size + 1 if closest_sector_index == 1 else valley[closest_sector_index] + self.max_size - 1
                    else:
                        k_f = valley[-1 * closest_sector_index + 1]

                    k_t = (k_f + valley[closest_sector_index]) / 2
                    return self.normalize_angle(math.radians(self.alpha * k_t))
            else:
                if target_sector >= start:
                    t_s = target_sector - start
                    t_e = (num_sectors - target_sector + end)
                else:
                    t_s = num_sectors - start + target_sector
                    t_e = end - target_sector

                if min(t_s, t_e) >=  self.min_size:
                    return beta_r
                elif t_s < self.min_size and valley_size >= 2 * self.min_size:
                    return self.normalize_angle(math.radians(self.alpha * ((start + self.min_size) % num_sectors)))
                elif t_e < self.min_size and valley_size >= 2 * self.min_size:
                    return self.normalize_angle(math.radians(self.alpha * ((num_sectors + end - self.min_size) % num_sectors)))
                else:
                    if valley_size > self.max_size:
                        k_f = (valley[closest_sector_index] + self.max_size - 1) % num_sectors if closest_sector_index == 0 else (valley[closest_sector_index] - self.max_size + 1 + num_sectors) % num_sectors
                    else:
                        k_f = valley[-1 * closest_sector_index + 1]
                    if k_f <= valley[1]:
                        k_t = (k_f + valley[closest_sector_index]) / 2 if closest_sector_index == 1 else (valley[closest_sector_index] + (num_sectors + k_f - valley[closest_sector_index]) / 2) % num_sectors
                    else:
                        k_t = (k_f + valley[closest_sector_index]) / 2 if closest_sector_index == 0 else (k_f + (num_sectors + valley[closest_sector_index] - k_f) / 2) % num_sectors
                    return self.normalize_angle(math.radians(self.alpha * k_t))
        else:
            if end >= start:
                if valley_size > self.max_size:
                    k_f = valley[closest_sector_index] - self.max_size + 1 if closest_sector_index == 1 else valley[closest_sector_index] + self.max_size - 1
                else:
                    k_f = valley[-1 * closest_sector_index + 1]
                k_t = (k_f + valley[closest_sector_index]) / 2
            else:
                if valley_size > self.max_size:
                    k_f = (valley[closest_sector_index] + self.max_size - 1) % num_sectors if closest_sector_index == 0 else (valley[closest_sector_index] - self.max_size + 1 + num_sectors) % num_sectors
                else:
                    k_f = valley[-1 * closest_sector_index + 1]
                if k_f <= valley[1]:
                    k_t = (k_f + valley[closest_sector_index]) / 2 if closest_sector_index == 1 else (valley[closest_sector_index] + (num_sectors + k_f - valley[closest_sector_index]) / 2) % num_sectors
                else:
                    k_t = (k_f + valley[closest_sector_index]) / 2 if closest_sector_index == 0 else (k_f + (num_sectors + valley[closest_sector_index] - k_f) / 2) % num_sectors

            return self.normalize_angle(math.radians(self.alpha * k_t))

    def transform_to_global_frame(self, base_frame_points):
        robot_x, robot_y = self.X
        robot_yaw = self.yaw

        global_points = []
        if base_frame_points:
            for base_x, base_y in base_frame_points:
                x_global = robot_x + base_x * math.cos(robot_yaw) - base_y * math.sin(robot_yaw)
                y_global = robot_y + base_x * math.sin(robot_yaw) + base_y * math.cos(robot_yaw)
                global_points.append((x_global, y_global))

        return global_points

    def odometry_callback(self, msg: Odometry):
        self.odometry_data = msg.pose.pose
        self.X = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, self.yaw = euler_from_quaternion(orientation_list)
    
    def lidar_callback(self, msg: LaserScan):
        """ Recieve and process the Lidar data """
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link',  
                'rplidar_link',  
                rclpy.time.Time()  
            )

            self.lidar_ranges = msg.ranges
            base_frame_points = []

            for i, range_value in enumerate(msg.ranges):
                if range_value >= msg.range_min and range_value <= min(msg.range_max, self.max_range):
                    angle = msg.angle_min + i * msg.angle_increment

                    point_in_lidar = PointStamped()
                    point_in_lidar.header.frame_id = 'rplidar_link'
                    point_in_lidar.point.x = range_value * math.cos(angle)
                    point_in_lidar.point.y = range_value * math.sin(angle)
                    point_in_lidar.point.z = 0.0

                    point_in_base = tf2_geometry_msgs.do_transform_point(point_in_lidar, transform)

                    
                    base_frame_points.append((point_in_base.point.x, point_in_base.point.y))

            self.lidar_data_base_frame = base_frame_points

        except Exception as e:
            self.get_logger().error(f"Failed to transform lidar points: {e}")
    
    """
    def save_execution_time_statistics(self):
        #Compute and save statistics for execution times.
        mean_time = statistics.mean(self.execution_times)
        stdev_time = statistics.stdev(self.execution_times)

        # Log statistics
        #elf.get_logger().info(f"Average Execution Time: {mean_time:.6f} seconds")
        #self.get_logger().info(f"Standard Deviation: {stdev_time:.6f} seconds")

        # Save to file
        with open('execution_time_stats_VFH.csv', mode='w', newline='') as file:
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
        #Start timer
        #start_time = time.time()
        if self.lidar_ranges is not None and self.odometry_data is not None:
            tester = True
            Xd = self.target_position
            X = self.X
            u_d = Xd - X
            phi = self.yaw
            if LA.norm(u_d) < 0.02:
                tester_2 = False
                l_velocity = 0.0
                a_velocity = 0.0

                self.save_execution_time_statistics()
                self.shutdown()

                
            elif LA.norm(u_d) < 0.5:
                phi_s = math.atan2(u_d[1], u_d[0])
                delta_phi = phi - phi_s
                if delta_phi <= (-math.pi):
                        delta_phi += 2 * math.pi
                if delta_phi > math.pi:
                        delta_phi -= 2 * math.pi
                l_velocity = self.v_gain * LA.norm(u_d)
                a_vel = - self.steering_gain * delta_phi
                a_velocity = np.sign(a_vel) * min(abs(a_vel), self.a_max)
               
            else:
                base_frame_points = self.lidar_data_base_frame
                global_points = self.transform_to_global_frame(base_frame_points)
                self.update_cartesian_histogram(global_points)
                polar_histogram = self.update_polar_histogram()
                valleys = self.candidate_valleys(polar_histogram)
                closest_valley, closest_sector_index, inside_valley = self.selected_valley(valleys)
                phi_s_0 = self.steering_angle(closest_sector_index, closest_valley, inside_valley)
                if phi_s_0 is None:
                    l_velocity = 0.0
                    a_velocity = 0.0
                else:
                    phi_s = self.normalize_angle2(phi_s_0)
                    delta_phi = phi - phi_s
                    if delta_phi <= (-math.pi):
                            delta_phi += 2 * math.pi
                    if delta_phi >= math.pi:
                        delta_phi -= 2 * math.pi
                    a_vel = - self.steering_gain * delta_phi
                    a_velocity = np.sign(a_vel) * min(abs(a_vel), self.a_max)
                    index = int(math.degrees(phi_s_0) / self.alpha)
                    h_1c = polar_histogram[index]
                    h_2c = min(h_1c, self.h_m)
                    l_velocity = min(0.8 * (1 - h_2c / self.h_m) * (1 - a_velocity / self.a_max) + self.v_min, self.v_max)
                    
             
        else:
            l_velocity = 0.0
            a_velocity = 0.0
        msg = Twist()
        msg.linear.x = l_velocity
        msg.angular.z = a_velocity
        self.publisher_.publish(msg)
        # End timer and record execution time
        #end_time = time.time()
        #loop_time = end_time - start_time
        #if tester and tester_2:
            #self.execution_times.append(loop_time)
        
            #file_path = "robot_trajectory_VFH.csv"

            #if not os.path.isfile(file_path):
                #with open(file_path, mode='w', newline='') as file:
                    #writer = csv.writer(file)
                    #writer.writerow(["X", "Y"]) 

            #with open(file_path, mode='a', newline='') as file:
                #writer = csv.writer(file)
                #writer.writerow([X[0], X[1]]) 



def main(args=None):
    rclpy.init(args=args)
    node = VFHNavigation()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()