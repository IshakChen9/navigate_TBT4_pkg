import math
#import time
#import statistics
from numpy import linalg as LA
import numpy as np
from shapely.geometry import Point, LineString, Polygon, GeometryCollection, MultiPolygon, MultiPoint
from shapely.ops import split
from scipy.spatial import Voronoi
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



class separatingHyperplaneNavigation(Node):
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

        # Initialize the attributes
        self.lidar_ranges = None 
        self.odometry_data = None 

        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Define necessary parameters for the LiDAR and obstacle detection

        # Initialize the TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Threshold distance between points to consider them part of the same obstacle
        self.obstacle_threshold = 0.1  

        # Angular difference threshold to consider them part of the same obstacle
        self.angular_threshold = math.radians(10)  

        # Minimum number of points required to consider something an obstacle
        self.min_points_per_obstacle = 5

        # Maximum range for the LIDAR
        self.max_range = 1.5  

        # Robot radius for dilation
        self.robot_radius = 0.3 

        # Define parameters for the control

        # Target's position
        self.target_position = np.array([6.6, -3])  

        # Define the gain of the nominal control Ud
        self.gain = 1
        self.max_l_vel = 0.31
        self.max_a_vel = 1.9


        self.workspace_bounds = Polygon([
                                            (-2, 2),  # Bottom-left corner
                                            (-2, -5),  # Bottom-right corner
                                            (8, -5),  # Top-right corner
                                            (8, 2),  # Top-left corner
                                        ])


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
                # Merge the obstacles
                merged_obstacle = {
                    'points': last_obstacle['points'] + first_obstacle['points'],
                    'start_angle': last_obstacle['start_angle'],
                    'end_angle': first_obstacle['end_angle']
                }
                obstacles = [merged_obstacle] + obstacles[1:-1]

        return obstacles


    def get_closest_point(self, obstacle):
        robot_center = Point(0, 0)
        points = [Point(q[0], q[1]) for q in obstacle['points']]
        closest_point = min(points, key=lambda p: robot_center.distance(p))
        return (closest_point.x * (1 + 0.3 / robot_center.distance(closest_point)), closest_point.y * (1 + 0.3 / robot_center.distance(closest_point)))  # Return as tuple

    def compute_perpendicular_bisector(self, obstacle_point):
        """Compute the perpendicular bisector between the robot center and an obstacle point."""
        robot_x, robot_y = 0, 0
        obstacle_x = obstacle_point[0]
        obstacle_y = obstacle_point[1]

        # Compute the midpoint
        mid_x = (robot_x + obstacle_x) / 2
        mid_y = (robot_y + obstacle_y) / 2

        # Compute the normal vector (pointing from the robot center to the midpoint)
        normal_x = mid_x - robot_x
        normal_y = mid_y - robot_y

        C = normal_x * mid_x + normal_y * mid_y

        # Generate the line as a LineString
        line = LineString([
            (mid_x - 1e6 * normal_y, mid_y + 1e6 * normal_x),  
            (mid_x + 1e6 * normal_y, mid_y - 1e6 * normal_x),  
        ])
        return line


    def split_workspace_with_bisector(self, bisector, workspace_boundary):
        """Split the workspace into two polygons using the bisector and return the polygon containing the robot"""
        robot_center = Point(0, 0)
        # Cut the workspace boundary with the bisector
        split_result = split(workspace_boundary, bisector)

        # Handle cases where the split doesn't produce two distinct regions
        if not isinstance(split_result, (GeometryCollection, Polygon)):  # Added Polygon to handle single-region splits
            raise ValueError("Workspace was not split correctly into two regions.")

        # Find the polygon containing the robot
        for region in split_result.geoms:
            if region.contains(robot_center):
                return region


    def compute_voronoi_cell(self):
        """Compute the Voronoi cell for the robot"""
        obstacles = self.detected_obstacles
        robot_center = Point(0, 0)
        workspace_boundary = self.transform_polygon(self.workspace_bounds, 2)
        # Case 0: No obstacles
        if len(obstacles) == 0:
            return workspace_boundary

        # Case 1: One obstacle
        if len(obstacles) == 1:
            closest_point = self.get_closest_point(obstacles[0])
            bisector = self.compute_perpendicular_bisector(closest_point)
            return self.split_workspace_with_bisector(bisector, workspace_boundary)

        # Case 2: Two obstacles
        if len(obstacles) == 2:
            closest_points = [self.get_closest_point(obs) for obs in obstacles]
            bisector1 = self.compute_perpendicular_bisector(closest_points[0])
            bisector2 = self.compute_perpendicular_bisector(closest_points[1])

            # Clip the workspace to the intersection of the half-spaces
            region1 = self.split_workspace_with_bisector(bisector1, workspace_boundary)
            region2 = self.split_workspace_with_bisector(bisector2, region1)
            return region2

        # Case 3: Three or more obstacles (use Voronoi diagram)
        else:
            points = [self.get_closest_point(obs) for obs in obstacles]
            points_coords = np.array([[p[0], p[1]] for p in points] + [[robot_center.x, robot_center.y]])
            vor = Voronoi(points_coords)

            # Find the region corresponding to the robot center
            robot_index = len(points_coords) - 1
            region_index = vor.point_region[robot_index]
            region = vor.regions[region_index]

            if -1 in region:
                return workspace_boundary  # Fallback for unbounded region

            # Construct Voronoi cell as a Polygon
            voronoi_cell = Polygon([vor.vertices[v] for v in region])
            return voronoi_cell.intersection(workspace_boundary)

    def local_freespace(self):
        """Compute the local free space of the robot"""
        robot_center = Point(0, 0)
        robot_radius = self.robot_radius
        Rs = self.max_range
        voronoi_cell = self.compute_voronoi_cell()
        circle = robot_center.buffer((robot_radius + Rs) / 2)
        local_workspace = voronoi_cell.intersection(circle)
        return local_workspace.buffer(-robot_radius)

    def robot_to_global_frame(self, point):
        """Transform point to from the robot frame to the global frame"""
        x_robot = self.X[0]
        y_robot = self.X[1]
        theta = self.yaw
        x_local, y_local = point
        x_global = x_robot + x_local * np.cos(theta) - y_local * np.sin(theta)
        y_global = y_robot + x_local * np.sin(theta) + y_local * np.cos(theta)
        return (x_global, y_global)
    
    def global_to_robot_frame(self, point):
        """Transform point to from the global frame to the robot frame"""
        gx, gy = point
        rx, ry = self.X
        theta = self.yaw
        translated_x = gx - rx
        translated_y = gy - ry
        rotation_matrix = np.array([
            [np.cos(-theta), -np.sin(-theta)],
            [np.sin(-theta),  np.cos(-theta)]
        ])
        robot_frame_point = np.dot(rotation_matrix, np.array([translated_x, translated_y]))
        return robot_frame_point
    
    def transform_polygon(self, polygon, test):
        """Transform the vertices a polygon from the robot frame to thr global frame if test == 1 and vice versa if test != 1"""
        if test == 1:
            transformed_coords = [self.robot_to_global_frame(coord) for coord in polygon.exterior.coords]
            return Polygon(transformed_coords)
        else:
            transformed_coords = [self.global_to_robot_frame(coord) for coord in polygon.exterior.coords]
            return Polygon(transformed_coords)


    def projection_omega(self):
        """Project the target onto the local free space considering the angular velocity constraint"""
        robot_center = Point(self.X[0], self.X[1])
        destination = Point(self.target_position[0], self.target_position[1])
        L_freespace = self.transform_polygon(self.local_freespace(), 1)
        if L_freespace.contains(destination):
            return destination
        else:
            Line_segment = LineString([
                (robot_center.x, robot_center.y),  
                (destination.x, destination.y),  
            ])
            polin = LineString(list(L_freespace.exterior.coords))
            return polin.intersection(Line_segment)

    def projection_v(self):
        """Project the target onto the local free space considering the linear velocity constraint"""
        robot_center = Point(self.X[0], self.X[1])
        phi = self.yaw
        destination = Point(self.target_position[0], self.target_position[1])
        L_freespace = self.transform_polygon(self.local_freespace(), 1)
        Line_segment = LineString([
            (robot_center.x, robot_center.y),  
            (robot_center.x + 1e6 * math.cos(phi), robot_center.y + 1e6 * math.sin(phi)),  
        ])
        polin = LineString(list(L_freespace.exterior.coords))
        if L_freespace.contains(destination):
            distance = LA.norm(self.X - self.target_position)
            return Point(robot_center.x + distance * math.cos(phi), robot_center.y + distance * math.sin(phi))
        else:
            return polin.intersection(Line_segment)
        

    def projection_l(self):
        """Compute the metric projection of the target onto the freespace"""
        polygon = self.transform_polygon(self.local_freespace(), 1)
        destination = Point(self.target_position[0], self.target_position[1])
        if polygon.contains(destination):
            return destination
        else:
            projected_point = polygon.exterior.interpolate(polygon.exterior.project(destination))
            return projected_point


    def linear_velocity(self,p_v):
        """Calculate the linear velocity"""
        gamma = self.gain
        theta = self.yaw
        position = self.X
        pv = np.array([p_v.x, p_v.y])
        v = np.array([math.cos(theta), math.sin(theta)])
        return gamma * np.dot(v, pv - position)

    def angular_velocity(self,p_l, p_w):
        """Calculate the angular velocity"""
        gamma = self.gain
        theta = self.yaw
        position = self.X
        pl = np.array([p_l.x, p_l.y])
        pw = np.array([p_w.x, p_w.y])
        v1 = np.array([math.cos(theta), math.sin(theta)])
        v2 = np.array([- math.sin(theta), math.cos(theta)])
        term1 = np.dot(v2, position - (pl + pw)/2)
        term2 = np.dot(v1, position - (pl + pw)/2)
        if term2 == 0:
            return 0
        else:
            return gamma * math.atan(term1 / term2)

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
            obstacles = []
            current_obstacle = []
            prev_point = None
            
            for i, range_value in enumerate(msg.ranges):
                if range_value >= msg.range_min and range_value <= min(msg.range_max, self.max_range):
                    angle = msg.angle_min + i * msg.angle_increment
                    point_in_lidar = PointStamped()
                    point_in_lidar.header.frame_id = 'rplidar_link'
                    point_in_lidar.point.x = range_value * math.cos(angle)
                    point_in_lidar.point.y = range_value * math.sin(angle)
                    point_in_lidar.point.z = 0.0

                    # Transform the point to the base frame
                    point_in_base = tf2_geometry_msgs.do_transform_point(point_in_lidar, transform)

                    # If this is the first point of a potential obstacle
                    if not current_obstacle:
                        current_obstacle.append((point_in_base.point.x, point_in_base.point.y))
                        prev_point = point_in_base
                        prev_angle = angle
                    else:
                        # Compute the distance to the previous point
                        distance = math.sqrt(
                            (point_in_base.point.x - prev_point.point.x) ** 2 +
                            (point_in_base.point.y - prev_point.point.y) ** 2
                        )


                        # Check if the current point belongs to the same obstacle
                        if distance <= self.obstacle_threshold:
                            # Add point to the current obstacle
                            current_obstacle.append((point_in_base.point.x, point_in_base.point.y))
                            prev_point = point_in_base
                            #prev_angle = current_angle
                            prev_angle = angle
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
                            #prev_angle = angle

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
        # Compute and save statistics for execution times.
        mean_time = statistics.mean(self.execution_times)
        stdev_time = statistics.stdev(self.execution_times)

        # Log statistics
        #self.get_logger().info(f"Average Execution Time: {mean_time:.6f} seconds")
        #self.get_logger().info(f"Standard Deviation: {stdev_time:.6f} seconds")

        # Save to file
        with open('execution_time_stats_SH.csv', mode='w', newline='') as file:
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
            if LA.norm(X - Xd) < 0.02:
                #tester_2 = False
                l_velocity = 0.0
                a_velocity = 0.0
                #self.save_execution_time_statistics()
                #self.shutdown()

            else:
                p_w = self.projection_omega()
                p_v = self.projection_v()
                p_l = self.projection_l()
                l_velocity = min(self.max_l_vel, self.linear_velocity(p_v))
                a_velocity = np.sign(self.angular_velocity(p_l, p_w)) * min(self.max_a_vel, abs(self.angular_velocity(p_l, p_w)))

        else:
            l_velocity = 0.0
            a_velocity = 0.0
        msg = Twist()
        msg.linear.x = l_velocity
        msg.angular.z = a_velocity
        self.publisher_.publish(msg)
        #end_time = time.time()
        #loop_time = end_time - start_time
        #if tester and tester_2:
            #self.execution_times.append(loop_time)

            #file_path = "robot_trajectory_SH.csv"

            #if not os.path.isfile(file_path):
                #with open(file_path, mode='w', newline='') as file:
                    #writer = csv.writer(file)
                    #writer.writerow(["X", "Y"])  

            #with open(file_path, mode='a', newline='') as file:
                #writer = csv.writer(file)
                #writer.writerow([X[0], X[1]]) 


def main(args=None):
    rclpy.init(args=args)
    node = separatingHyperplaneNavigation()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

