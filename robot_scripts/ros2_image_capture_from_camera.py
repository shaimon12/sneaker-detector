import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ImageCaptureNode(Node):
    def __init__(self):
        super().__init__('image_capture_node')
        self.bridge = CvBridge()
        self.latest_frame = None
        self.image_count = 1

        # Setup subscription to camera topic
        self.subscription = self.create_subscription(
            Image,
            '/depth_cam/rgb/image_raw',
            self.image_callback,
            10
        )

        # Setup save directory
        self.save_dir = os.path.join(os.path.dirname(__file__), 'captured_images')
        os.makedirs(self.save_dir, exist_ok=True)

        self.get_logger().info("Press ENTER in the OpenCV window to save image. ESC to exit.")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_frame = frame
            cv2.imshow("Camera Feed - Press Enter to Save", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 13:  # Enter key
                self.save_image()
            elif key == 27:  # Esc key
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")

    def save_image(self):
        if self.latest_frame is not None:
            filename = f"_{self.image_count}.jpg"
            filepath = os.path.join(self.save_dir, filename)
            cv2.imwrite(filepath, self.latest_frame)
            self.get_logger().info(f"Saved image: {filepath}")
            self.image_count += 1
        else:
            self.get_logger().warn("No frame to save.")

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ImageCaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
