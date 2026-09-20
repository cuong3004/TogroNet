import rclpy
import cv2
import os

from rclpy.node import Node
from std_msgs.msg import String

from ultralytics import YOLO
import json


class DetectorNode(Node):

    def __init__(self):

        super().__init__('detector_node')

        self.model = YOLO("tomato_model.pt")

        self.dataset = os.path.expanduser("~/tomato_dataset")

        self.result_pub = self.create_publisher(
            String,
            '/detection_result',
            10
        )

        self.create_subscription(
            String,
            '/capture_done',
            self.detect,
            10
        )


    def detect(self,msg):

        plant_id = msg.data

        path = f"{self.dataset}/{plant_id}.jpg"

        img = cv2.imread(path)

        detections = self.model(img)[0]

        tomato = 0
        ripe = 0
        flower = 0

        for box in detections.boxes:

            cls = int(box.cls)

            if cls == 0:
                tomato += 1

            elif cls == 1:
                ripe += 1

            elif cls == 2:
                flower += 1

        result = {

            "plant_id":plant_id,
            "tomato":tomato,
            "ripe":ripe,
            "flower":flower
        }

        msg_out = String()

        msg_out.data = json.dumps(result)

        self.result_pub.publish(msg_out)



def main():

    rclpy.init()

    node = DetectorNode()

    rclpy.spin(node)

    rclpy.shutdown()