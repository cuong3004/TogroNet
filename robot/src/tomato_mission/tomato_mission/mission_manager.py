import rclpy
import yaml
import os

from rclpy.node import Node
from rclpy.action import ActionClient

from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String


class MissionManager(Node):

    def __init__(self):

        super().__init__('mission_manager')

        self.declare_parameter('plants')
        self.declare_parameter('retry_navigation',3)

        plant_file = self.get_parameter('plants').value

        with open(plant_file,'r') as f:
            self.plants = yaml.safe_load(f)['plants']

        self.retry = self.get_parameter('retry_navigation').value

        self.nav_client = ActionClient(self,NavigateToPose,'navigate_to_pose')

        self.capture_pub = self.create_publisher(String,'/plant_reached',10)

        self.current_index = 0

        self.run_mission()


    def navigate(self,x,y, z):

        goal = NavigateToPose.Goal()

        pose = PoseStamped()

        pose.header.frame_id = "map"

        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z = z
        pose.pose.orientation.w = 1.0

        goal.pose = pose

        self.nav_client.wait_for_server()

        future = self.nav_client.send_goal_async(goal)

        rclpy.spin_until_future_complete(self,future)

        goal_handle = future.result()

        if not goal_handle.accepted:
            return False

        result_future = goal_handle.get_result_async()

        rclpy.spin_until_future_complete(self,result_future)

        status = result_future.result().status

        return status == 4


    def run_mission(self):

        for plant in self.plants:

            plant_id = plant['id']
            x = plant['x']
            y = plant['y']
            z = plant['z']

            success = False

            for i in range(self.retry):

                self.get_logger().info(f"Going to {plant_id} try {i}")

                if self.navigate(x,y,z):

                    success = True
                    break

            if not success:

                self.get_logger().error(f"Skip {plant_id}")
                continue

            msg = String()

            msg.data = plant_id

            self.capture_pub.publish(msg)

            rclpy.spin_once(self,timeout_sec=3)



def main():

    rclpy.init()

    node = MissionManager()

    rclpy.spin(node)

    rclpy.shutdown()