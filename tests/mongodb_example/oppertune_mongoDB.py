import os
import subprocess
import time
import csv
import socket
import re
import logging
from typing import Any, Mapping, Union

from pymongo import MongoClient
from oppertune.algorithms.hybrid_solver import HybridSolver
from oppertune.core.values import Integer, Categorical

logging.basicConfig(
    level=logging.INFO,
    filename="tuning.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

ParameterTypes = Union[Integer, Categorical]


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
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
    def _kill_flask():
        subprocess.run(
            "lsof -ti :5000 2>/dev/null | xargs -r kill -9",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(["pkill", "-f", "web_server.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @staticmethod
    def is_port_in_use(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", port)) == 0

    @staticmethod
    def parse_wrk_output(text: str) -> Mapping[str, float]:
        tp, lat = 0.0, 1000.0
        for line in text.splitlines():
            if m := re.search(r"Requests/sec:\s*([\d.]+)", line):
                tp = float(m.group(1))
            if m := re.search(r"Latency\s+([\d.]+)(us|ms|s)", line):
                v, u = float(m.group(1)), m.group(2)
                lat = v / 1000 if u == "us" else v if u == "ms" else v * 1000
        return {"throughput": tp, "latency": lat}

    def _wait_for_mongo(self, port: int, timeout: int = 20) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.is_port_in_use(port):
                time.sleep(0.2)
                continue
            try:
                MongoClient(
                    f"mongodb://127.0.0.1:{port}/", serverSelectionTimeoutMS=500
                ).server_info()
                return True
            except Exception:
                time.sleep(0.3)
        return False

    def _wait_for_flask(self, timeout: int = 10):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.FLASK_PORT), 1):
                    return True
            except OSError:
                time.sleep(0.2)
        raise TimeoutError("Flask did not start")

    def _write_config(self, params: Mapping[str, Any], mongo_port: int):
        with open(self.mongo_conf_path, "w") as f:
            f.write(
                f"""
storage:
  dbPath: /tmp/mongodb_data
  journal:
    enabled: true
    commitIntervalMs: {params['journalCommitInterval']}
  wiredTiger:
    engineConfig:
      cacheSizeGB: {max(params['cache_size_mb'] / 1024, 0.25):.2f}
net:
  port: {mongo_port}
  bindIp: {params['net.bindIp']}
  maxIncomingConnections: {params['maxConns']}
operationProfiling:
  mode: {params['operationProfiling.mode']}
  slowOpThresholdMs: {params['operationProfiling.slowOpThresholdMs']}
setParameter:
  logLevel: {params['logLevel']}
  rangeDeleterBatchDelayMS: {params['setParameter.rangeDeleterBatchDelayMS']}
  rangeDeleterBatchSize: {params['setParameter.rangeDeleterBatchSize']}
  transactionLifetimeLimitSeconds: {params['setParameter.transactionLifetimeLimitSeconds']}
"""
            )

    def run(self, params: Mapping[str, Any]):
        if self.mongo_proc:
            self.mongo_proc.terminate(); self.mongo_proc.wait()
        subprocess.run(["pkill", "mongod"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._kill_flask()

        os.makedirs("/tmp/mongodb_data", exist_ok=True)
        open(self.mongo_log_path, "w").close()

        mongo_port = pick_free_port()
        self._write_config(params, mongo_port)

        self.mongo_proc = subprocess.Popen([
            "mongod", "--config", self.mongo_conf_path,
            "--logpath", self.mongo_log_path, "--logappend"
        ])
        if not self._wait_for_mongo(mongo_port):
            self.metrics = {"throughput": 0.0, "latency": 1000.0}; return

        self.web_proc = subprocess.Popen([
            "python3", "web_server.py"], env={**os.environ, "MONGO_PORT": str(mongo_port)}
        )
        try:
            self._wait_for_flask()
        except Exception:
            self.metrics = {"throughput": 0.0, "latency": 1000.0}; return

        try:
            base = f"http://127.0.0.1:{self.FLASK_PORT}"
            subprocess.run(["curl", "-s", f"{base}/set?key=test&val=123"], check=True)
            out = subprocess.check_output(["wrk", "-t2", "-c50", "-d5s", f"{base}/get?key=test"], text=True)
            self.metrics = self.parse_wrk_output(out)
        except subprocess.CalledProcessError:
            self.metrics = {"throughput": 0.0, "latency": 1000.0}

    def get_metrics(self) -> Mapping[str, float]:
        return self.metrics

    def cleanup(self):
        if self.web_proc:
            self.web_proc.terminate(); self.web_proc.wait()
        if self.mongo_proc:
            self.mongo_proc.terminate(); self.mongo_proc.wait()
        subprocess.run(["pkill", "mongod"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._kill_flask()


def calculate_reward(m: Mapping[str, Any]) -> float:
    return m["throughput"] / (m["latency"] + 1)


def main():
    app = Application()

    params_to_tune = [
        Integer("cache_size_mb", 512, 256, 4096, 64),
        Categorical("net.bindIp", "127.0.0.1", ("127.0.0.1",)),
        Categorical("operationProfiling.mode", "off", ("off", "slowOp", "all")),
        Integer("journalCommitInterval", 100, 5, 500, 5),
        Integer("maxConns", 1000, 500, 20000, 500),
        Integer("operationProfiling.slowOpThresholdMs", 100, 50, 1000, 50),
        Integer("setParameter.rangeDeleterBatchDelayMS", 100, 0, 1000, 20),
        Integer("setParameter.rangeDeleterBatchSize", 500, 100, 100000, 100),
        Integer("setParameter.transactionLifetimeLimitSeconds", 60, 60, 3600, 60),
        Categorical("logLevel", "0", ("0", "1", "2")),
    ]

    solver = HybridSolver(
        params_to_tune,
        categorical_algorithm="exponential_weights_slates",
        categorical_algorithm_args={"random_seed": 42},
        numerical_algorithm="bluefin",
        numerical_algorithm_args={"feedback": 2, "eta": 0.01, "delta": 0.1, "random_seed": 42},
    )

    with open("opper_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "throughput", "latency", "reward"] + [p.name for p in params_to_tune])
        writer.writeheader()

        try:
            for it in range(300):
                pred, req_id = solver.predict()
                logging.info(f"[{it}] params: {pred}")
                print(f"[{it}] params: {pred}")

                app.run(pred)
                metrics = app.get_metrics()
                reward = calculate_reward(metrics)
                logging.info(f"reward: {reward:.4f}")
                print(f"Metrics: {metrics}, Reward: {reward:.4f}")

                solver.store_reward(req_id, reward)
                solver.set_reward(reward)

                writer.writerow({
                    "iteration": it,
                    "throughput": metrics["throughput"],
                    "latency": metrics["latency"],
                    "reward": reward,
                    **pred
                })
        finally:
            app.cleanup()


if __name__ == "__main__":
    main()
