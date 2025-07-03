import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
from cv_bridge import CvBridge
import torch
import torch.nn as nn
import numpy as np
from torchvision import models

# === Sneaker Class Labels ===
SNEAKER_CLASSES = [
    "Nike_Infinity_React_3",  # Forward
    "Nike_Vomero_17",         # Left
    "Nike_Winflo_10"          # Right
]

# === Action Map for Each Class ===
SNEAKER_ACTIONS = {
    "Nike_Infinity_React_3": (0.2, 0.0, "Moving Forward"),
    "Nike_Vomero_17": (0.0, 0.3, "Turning Left"),
    "Nike_Winflo_10": (0.0, -0.3, "Turning Right"),
    "NO_OBJECT": (0.0, 0.0, "No object detected - Staying Still"),
    "EMERGENCY_STOP": (0.0, 0.0, "EMERGENCY STOPPED (Spacebar)")
}

class SneakerDetector(Node):
    def __init__(self):
        super().__init__('sneaker_detector')

        # === Model Setup ===
        self.model_path = '/home/ubuntu/inceptionv3_robot_final_pytorch_3_unfrozen.pth'  # Update path as needed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = 0.5
        self.bridge = CvBridge()
        self.emergency_stop = False

        # === Load InceptionV3 Model ===
        self.model = models.inception_v3(aux_logits=False)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, len(SNEAKER_CLASSES))
        )
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # === ROS Topics ===
        self.subscription = self.create_subscription(Image, '/depth_cam/rgb/image_raw', self.image_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info("Sneaker detection node initialized.")

        self.twist = Twist()

    def preprocess(self, img):
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # Normalize as in training
        img = img.transpose(2, 0, 1)
        return torch.tensor(img).unsqueeze(0).to(self.device)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.preprocess(img)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            conf = conf.item()
            label = SNEAKER_CLASSES[pred.item()] if conf >= self.confidence_threshold else "NO_OBJECT"

        if self.emergency_stop:
            label = "EMERGENCY_STOP"

        # === Execute Action ===
        lin, ang, desc = SNEAKER_ACTIONS.get(label, SNEAKER_ACTIONS["NO_OBJECT"])
        self.twist.linear.x = lin
        self.twist.angular.z = ang
        self.cmd_pub.publish(self.twist)

        # === Display Overlay ===
        cv2.putText(frame, f"{label} ({conf:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Action: {desc}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Sneaker Detection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC to quit
            rclpy.shutdown()
        elif key == 32:  # Spacebar to toggle emergency stop
            self.emergency_stop = not self.emergency_stop

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SneakerDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
