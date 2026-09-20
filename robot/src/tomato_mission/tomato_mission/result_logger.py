import rclpy
import json
import os

from rclpy.node import Node
from std_msgs.msg import String


class ResultLogger(Node):

    def __init__(self):

        super().__init__('result_logger')

        self.result_file = os.path.expanduser("~/tomato_results.json")

        self.results = {}

        self.create_subscription(
            String,
            '/detection_result',
            self.save,
            10
        )


    def save(self,msg):

        data = json.loads(msg.data)

        plant_id = data["plant_id"]

        self.results[plant_id] = data

        with open(self.result_file,'w') as f:
            json.dump(self.results,f,indent=2)

        self.get_logger().info(f"Saved result for {plant_id}")



def main():

    rclpy.init()

    node = ResultLogger()

    rclpy.spin(node)

    rclpy.shutdown()