#!/usr/bin/env python3
import os
import subprocess
import time
import csv
import socket
import re
import logging
from typing import Any, Mapping, Union

from pymongo import MongoClient
from oppertune.algorithms.moe_ppo_tuner import MoEPPOAlgorithm
from oppertune.core.values import Integer, Categorical

logging.basicConfig(level=logging.INFO,
                    filename="tuning.log",
                    filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")

ParameterTypes = Union[Integer, Categorical]


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


class Application:
    FLASK_PORT = 5000

    def __init__(self):
        self.web_proc = None
        self.mongo_proc = None
        self.mongo_conf_path = "/tmp/mongod.conf"
        self.mongo_log_path = "/tmp/mongodb.log"
        self.metrics = {"throughput": 0.0, "latency": 1000.0}

    @staticmethod
    def _kill_flask_on_port(port: int = 5000):
        subprocess.run(f"lsof -ti :{port} 2>/dev/null | xargs -r kill -9",
                       shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["pkill", "-f", "web_server.py"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def wait_for_flask_ready(self, timeout: int = 10):
        for _ in range(timeout * 10):
            try:
                with socket.create_connection(("127.0.0.1", self.FLASK_PORT), 1):
                    return
            except OSError:
                time.sleep(0.1)
        raise TimeoutError("Flask server did not start in time")

    @staticmethod
    def _print_log_tail(path: str):
        try:
            with open(path) as f:
                tail = "".join(f.readlines()[-100:])
                logging.error("MongoDB log tail:\n" + tail)
                print("\n===== MongoDB log tail =====\n" + tail + "============================\n")
        except FileNotFoundError:
            logging.error("MongoDB log not found")

    @staticmethod
    def wait_for_mongo_ready(port: int, mongo_log_path: str, timeout: int = 20) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not Application.is_port_in_use(port):
                time.sleep(0.2)
                continue
            try:
                MongoClient(f"mongodb://127.0.0.1:{port}/",
                            serverSelectionTimeoutMS=500).server_info()
                return True
            except Exception:
                if not Application.is_port_in_use(port):
                    break
                time.sleep(0.2)
        Application._print_log_tail(mongo_log_path)
        return False

    def set_parameters(self, parameters: Mapping[str, Any], port: int):
        with open(self.mongo_conf_path, "w") as f:
            f.write(f"""\
storage:
  dbPath: /tmp/mongodb_data
  journal:
    enabled: true
    commitIntervalMs: {parameters['journalCommitInterval']}
  wiredTiger:
    engineConfig:
      cacheSizeGB: {max(parameters['cache_size_mb'] / 1024, 0.25):.2f}
net:
  port: {port}
  bindIp: {parameters['net.bindIp']}
  maxIncomingConnections: {parameters['maxConns']}
operationProfiling:
  mode: {parameters['operationProfiling.mode']}
  slowOpThresholdMs: {parameters['operationProfiling.slowOpThresholdMs']}
setParameter:
  logLevel: {parameters['logLevel']}
  rangeDeleterBatchDelayMS: {parameters['setParameter.rangeDeleterBatchDelayMS']}
  rangeDeleterBatchSize: {parameters['setParameter.rangeDeleterBatchSize']}
  transactionLifetimeLimitSeconds: {parameters['setParameter.transactionLifetimeLimitSeconds']}
""")
        logging.info(f"Generated MongoDB config (port {port}): {parameters}")

    def run(self, parameters: Mapping[str, Any]):
        port = pick_free_port()

        if os.path.exists(self.mongo_log_path):
            os.remove(self.mongo_log_path)
        if self.mongo_proc:
            self.mongo_proc.terminate()
            self.mongo_proc.wait()
        subprocess.run(["pkill", "mongod"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1)

        os.makedirs("/tmp/mongodb_data", exist_ok=True)
        open(self.mongo_log_path, "a").close()

        self.set_parameters(parameters, port)

        self.mongo_proc = subprocess.Popen(
            ["mongod", "--config", self.mongo_conf_path,
             "--logpath", self.mongo_log_path, "--logappend"]
        )

        if not self.wait_for_mongo_ready(port, self.mongo_log_path):
            self.metrics = {"throughput": 0.0, "latency": 1000.0}
            return

        self._kill_flask_on_port(self.FLASK_PORT)
        if self.web_proc:
            self.web_proc.terminate()
            self.web_proc.wait()

        self.web_proc = subprocess.Popen(
            ["python3", "web_server.py"],
            env={**os.environ, "MONGO_PORT": str(port)}
        )
        try:
            self.wait_for_flask_ready()
        except TimeoutError:
            self.metrics = {"throughput": 0.0, "latency": 1000.0}
            return

        try:
            base = f"http://127.0.0.1:{self.FLASK_PORT}"
            subprocess.run(["curl", "-s", f"{base}/set?key=test&val=123"],
                           check=True, capture_output=True)
            result = subprocess.run(
                ["wrk", "-t2", "-c50", "-d5s", f"{base}/get?key=test"],
                capture_output=True, text=True, check=True).stdout
        except subprocess.CalledProcessError as e:
            logging.error(f"Benchmark failed: {e}")
            self.metrics = {"throughput": 0.0, "latency": 1000.0}
            return

        self.metrics = self.parse_wrk_output(result)
        logging.info(f"Metrics: {self.metrics}")

    @staticmethod
    def is_port_in_use(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", port)) == 0

    @staticmethod
    def parse_wrk_output(output: str) -> Mapping[str, float]:
        throughput, latency = 0.0, 1000.0
        for line in output.splitlines():
            if m := re.search(r"Requests/sec:\s*([\d.]+)", line):
                throughput = float(m.group(1))
            if m := re.search(r"Latency\s+([\d.]+)(us|ms|s)", line):
                v, u = float(m.group(1)), m.group(2)
                latency = v / 1000 if u == "us" else v if u == "ms" else v * 1000
        return {"throughput": throughput, "latency": latency}

    def get_metrics(self) -> Mapping[str, float]:
        return self.metrics

    def cleanup(self):
        if self.mongo_proc:
            self.mongo_proc.terminate()
            self.mongo_proc.wait()
        if self.web_proc:
            self.web_proc.terminate()
            self.web_proc.wait()
        subprocess.run(["pkill", "mongod"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._kill_flask_on_port(self.FLASK_PORT)
        logging.info("Cleaned up processes")


def calculate_reward(metrics: Mapping[str, Any]) -> float:
    return metrics["throughput"] / (metrics["latency"] + 1)


def main():
    app = Application()

    parameters_to_tune = [
        {"param": Integer("cache_size_mb", 512, 256, 4096, 64), "expert_type": "CPU"},
        {"param": Categorical("net.bindIp", "127.0.0.1", ("127.0.0.1",)), "expert_type": "NET"},
        {"param": Categorical("operationProfiling.mode", "off", ("off", "slowOp", "all")), "expert_type": "CPU"},
        {"param": Integer("journalCommitInterval", 100, 5, 500, 5), "expert_type": "IO"},
        {"param": Integer("maxConns", 1000, 500, 20000, 500), "expert_type": "NET"},
        {"param": Integer("operationProfiling.slowOpThresholdMs", 100, 50, 1000, 50), "expert_type": "CPU"},
        {"param": Integer("setParameter.rangeDeleterBatchDelayMS", 100, 0, 1000, 20), "expert_type": "IO"},
        {"param": Integer("setParameter.rangeDeleterBatchSize", 500, 100, 100000, 100), "expert_type": "IO"},
        {"param": Integer("setParameter.transactionLifetimeLimitSeconds", 60, 60, 3600, 60), "expert_type": "CPU"},
        {"param": Categorical("logLevel", "0", ("0", "1", "2")), "expert_type": "CPU"},
    ]

    tuner = MoEPPOAlgorithm(parameters_to_tune)

    with open("btune_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "throughput", "latency", "reward"])
        writer.writeheader()

        try:
            for iteration in range(200):
                predicted_params = tuner.predict()
                logging.info(f"[{iteration}] params: {predicted_params}")
                print(f"[{iteration}] params: {predicted_params}")

                app.run(predicted_params)

                metrics = app.get_metrics()
                reward = calculate_reward(metrics)
                logging.info(f"reward: {reward:.4f}")
                print(f"reward: {reward:.4f}")

                tuner.set_reward(reward, metrics)

                writer.writerow({"iteration": iteration,
                                 "throughput": metrics["throughput"],
                                 "latency": metrics["latency"],
                                 "reward": reward})
        finally:
            app.cleanup()


if __name__ == "__main__":
    main()
