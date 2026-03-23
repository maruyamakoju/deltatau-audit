# ROS2 Integration

`deltatau-audit` provides native support for ROS2 (Robot Operating System) to bridge the gap between simulation and high-stakes physical robotics.

## The Time-Aware Agent Node

In a real robot, sensor data arrives asynchronously. Standard RL agents assume a fixed $\Delta t$, which leads to catastrophic failure if the network or message bus jitters.

Our `TimeAwareAgentNode` explicitly tracks the interval since the last valid observation and feeds it to the agent's internal clock.

### Usage

```python
from deltatau_audit.bridge.ros2 import TimeAwareAgentNode
from rclpy.executors import MultiThreadedExecutor

# ... initialize your adapter ...
node = TimeAwareAgentNode("walking_controller", adapter)
executor = MultiThreadedExecutor()
executor.add_node(node)
executor.spin()
```

## Hardware Latency Compensation

By using the **Temporal World Model**, the ROS2 node can "hallucinate" the state of the robot at the exact moment the command will reach the motors, effectively zeroing out the inference-to-actuation lag.
