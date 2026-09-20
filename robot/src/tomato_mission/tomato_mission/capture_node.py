import rclpy
import cv2
import os

from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from cv_bridge import CvBridge


class CaptureNode(Node):

    def __init__(self):

        super().__init__('capture_node')

        self.bridge = CvBridge()

        self.frame = None

        self.dataset = os.path.expanduser("~/tomato_dataset")

        os.makedirs(self.dataset,exist_ok=True)

        self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.create_subscription(
            String,
            '/plant_reached',
            self.capture,
            10
        )

        self.done_pub = self.create_publisher(
            String,
            '/capture_done',
            10
        )


    def image_callback(self,msg):

        self.frame = self.bridge.imgmsg_to_cv2(msg,'bgr8')


    def capture(self,msg):

        if self.frame is None:
            return

        plant_id = msg.data

        img = self.frame.copy()

        h,w,_ = img.shape

        crop = img[
            int(h*0.3):int(h*0.7),
            int(w*0.3):int(w*0.7)
        ]

        path = f"{self.dataset}/{plant_id}.jpg"

        cv2.imwrite(path,crop)

        self.get_logger().info(f"Saved {path}")

        done = String()

        done.data = plant_id

        self.done_pub.publish(done)



def main():

    rclpy.init()

    node = CaptureNode()

    rclpy.spin(node)

    rclpy.shutdown()