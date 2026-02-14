import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_system_default
from geometry_msgs.msg import *


class PoseArrayTester(Node):
    def __init__(self):
        super().__init__("pose_array_tester")

        self.pub_posearray = self.create_publisher(
            PoseArray, "/bezier_curve", qos_profile=qos_profile_system_default
        )

    def publish_posearray(self, pose_array: PoseArray):
        self.pub_posearray.publish(pose_array)


def main(args=None):
    rclpy.init(args=args)
    node = PoseArrayTester()

    try:
        r = node.create_rate(10.0)  # 10Hz
        while rclpy.ok():
            rclpy.spin_once(node)
            # Here you can create and publish a PoseArray for testing
            pose_array = PoseArray()
            pose_array.header.stamp = node.get_clock().now().to_msg()
            pose_array.header.frame_id = "world"
            # Add some test poses
            for i in range(5):
                pose = Pose()
                pose.position.x = float(i)
                pose.position.y = float(i * 2)
                pose.position.z = float(i * 3)
                pose_array.poses.append(pose)
            node.publish_posearray(pose_array)
            r.sleep()

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
