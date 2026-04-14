"""
Remote Hardware-in-the-loop (HIL) Bridge.

Enables auditing of physical robots located anywhere in the world
via a secure network socket. The audit engine runs locally, sending
actions and receiving sensor data from the remote hardware.
"""

import pickle
import socket
import time
from typing import Any

import gymnasium as gym


class RemoteRobotEnv(gym.Env):
    """
    Gym environment that acts as a client to a remote physical robot.
    Connects to a 'deltatau-server' running on the robot's onboard computer.
    """

    def __init__(self, host: str, port: int = 5005, timeout: float = 1.0):
        super().__init__()
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None

        # Space definitions (must be synced with remote robot)
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(12,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,))

    def connect(self):
        print(f"📡 Connecting to remote robot at {self.host}:{self.port}...")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.host, self.port))
        print("✅ Connection established.")

    def _send_recv(self, data: Any) -> Any:
        msg = pickle.dumps(data)
        self.sock.sendall(len(msg).to_bytes(4, byteorder="big"))
        self.sock.sendall(msg)

        header = self.sock.recv(4)
        length = int.from_bytes(header, byteorder="big")
        resp = b""
        while len(resp) < length:
            chunk = self.sock.recv(length - len(resp))
            if not chunk:
                break
            resp += chunk
        return pickle.loads(resp)

    def reset(self, seed=None, options=None):
        if self.sock is None:
            self.connect()
        # Request reset from robot
        obs, info = self._send_recv({"cmd": "reset"})
        return obs, info

    def step(self, action):
        # Send action to physical robot and wait for observation
        start_time = time.time()
        obs, reward, term, trunc, info = self._send_recv({"cmd": "step", "action": action})

        # Measure real-world transport latency
        info["network_latency_ms"] = (time.time() - start_time) * 1000
        return obs, reward, term, trunc, info

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None


class RemoteRobotServer:
    """
    Server to be run ON the robot hardware.
    Relays commands from the auditor to the physical actuators.
    """

    def __init__(self, local_robot_interface: Any, port: int = 5005):
        self.robot = local_robot_interface
        self.port = port

    def run(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("0.0.0.0", self.port))
        server.listen(1)
        print(f"🤖 Robot HIL Server listening on port {self.port}...")

        while True:
            conn, addr = server.accept()
            print(f"🔌 Auditor connected from {addr}")
            try:
                while True:
                    header = conn.recv(4)
                    if not header:
                        break
                    length = int.from_bytes(header, byteorder="big")
                    data = b""
                    while len(data) < length:
                        data += conn.recv(length - len(data))

                    req = pickle.loads(data)
                    if req["cmd"] == "reset":
                        resp = self.robot.reset()
                    elif req["cmd"] == "step":
                        resp = self.robot.step(req["action"])

                    msg_resp = pickle.dumps(resp)
                    conn.sendall(len(msg_resp).to_bytes(4, byteorder="big"))
                    conn.sendall(msg_resp)
            except Exception as e:
                print(f"❌ Error: {e}")
            finally:
                conn.close()
