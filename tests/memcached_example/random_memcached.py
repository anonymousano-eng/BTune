from typing import Any, Mapping, List, Union
import subprocess
import time
import csv
import os
import random
from oppertune.core.values import Integer, Real, Categorical

ParameterTypes = Union[Integer, Real, Categorical]

class Application:
    def __init__(self):
        self.web_proc = None

    def set_parameters(self, parameters: Mapping[str, Any]):
        self.threads = parameters["threads"]
        self.conn_limit = parameters["conn_limit"]
        self.port_TCP = parameters["port_TCP"]
        self.udp_port = parameters["udp_port"]
        self.memory = parameters["memory"]
        self.max_reqs_per_event = parameters["max_reqs_per_event"]
        self.slab_growth_factor = parameters["slab_growth_factor"]
        self.unix_mask = parameters["unix_mask"]
        self.lock_memory = parameters["lock_memory"]
        self.large_pages = parameters["large_pages"]
        self.disable_cas = parameters["disable_cas"]
        self.disable_flush = parameters["disable_flush"]
        self.verbose = parameters["verbose"]
        self.listen_backlog = parameters["listen_backlog"]

    def run(self):
        subprocess.run(["pkill", "memcached"])
        cmd = [
            "memcached",
            "-u", "nobody",
            "-t", str(self.threads),
            "-c", str(self.conn_limit),
            "-p", str(self.port_TCP),
            "-U", str(self.udp_port),
            "-m", str(self.memory),
            "-R", str(self.max_reqs_per_event),
            "-f", str(self.slab_growth_factor),
            "-a", self.unix_mask,
            "-b", str(self.listen_backlog),
            "-v", str(self.verbose),
        ]
        if self.lock_memory == "on":
            cmd.append("-k")
        if self.large_pages == "on":
            cmd.append("-L")
        if self.disable_cas == "on":
            cmd.append("-D")
        if self.disable_flush == "on":
            cmd.append("-F")
        subprocess.Popen(cmd)
        time.sleep(2)

        if self.web_proc:
            self.web_proc.terminate()
        self.web_proc = subprocess.Popen(["python3", "web_server.py"],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        subprocess.run(["curl", "http://localhost:5000/set?key=test&val=123"])

        result = subprocess.run(
            ["wrk", "-t2", "-c50", "-d5s", "http://localhost:5000/get?key=test"],
            capture_output=True, text=True
        ).stdout
        subprocess.run(["pkill", "memcached"])
        time.sleep(1)

        throughput = 0.0
        latency = 0.0
        for line in result.splitlines():
            if "Requests/sec" in line:
                throughput = float(line.split(":")[1].strip())
            if "Latency" in line and "Thread" not in line:
                latency_str = line.split()[1]
                try:
                    if "us" in latency_str:
                        latency = float(latency_str.replace("us", "")) / 1000
                    elif "ms" in latency_str:
                        latency = float(latency_str.replace("ms", ""))
                    elif "s" in latency_str:
                        latency = float(latency_str.replace("s", "")) * 1000
                except ValueError:
                    latency = 1000
        self.metrics = {"throughput": throughput, "latency": latency}

    def get_metrics(self):
        return self.metrics

def random_search(parameters_to_tune: List[ParameterTypes]) -> dict:
    param_dict = {}
    for param in parameters_to_tune:
        if isinstance(param, Integer):
            param_dict[param.name] = random.randint(param.min, param.max)
        elif isinstance(param, Real):
            param_dict[param.name] = round(random.uniform(param.min, param.max), 4)
        elif isinstance(param, Categorical):
            param_dict[param.name] = random.choice(param.categories)
    return param_dict

def calculate_reward(metrics: Mapping[str, Any]) -> float:
    throughput = metrics.get("throughput", 0)
    latency = metrics.get("latency", 1000)
    return (throughput / (latency + 1)) / 100

def main() -> None:
    app = Application()
    parameters_to_tune = [
        Integer("threads", val=4, min=1, max=16),
        Integer("conn_limit", val=1024, min=128, max=8192),
        Integer("port_TCP", val=11211, min=1, max=65535),
        Integer("udp_port", val=0, min=0, max=65535),
        Integer("memory", val=64, min=1, max=32768),
        Integer("max_reqs_per_event", val=20, min=10, max=100),
        Real("slab_growth_factor", val=1.25, min=1.0, max=2.0),
        Categorical("unix_mask", val="0700", categories=("0700", "0755", "0777")),
        Categorical("lock_memory", val="off", categories=("on", "off")),
        Categorical("large_pages", val="off", categories=("on", "off")),
        Integer("listen_backlog", val=1024, min=10, max=5000),
        Categorical("disable_cas", val="off", categories=("on", "off")),
        Categorical("disable_flush", val="off", categories=("on", "off")),
        Integer("verbose", val=1, min=0, max=3),
    ]
    with open("hybrid_results_random.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "throughput", "latency", "reward"])
        writer.writeheader()
        for iteration in range(10):
            predicted_params = random_search(parameters_to_tune)
            print(f"[{iteration}] Prediction: {predicted_params}")
            app.set_parameters(predicted_params)
            app.run()
            metrics = app.get_metrics()
            reward = calculate_reward(metrics)
            print(f"Reward: {reward:.4f}")
            writer.writerow({
                "iteration": iteration,
                "throughput": metrics["throughput"],
                "latency": metrics["latency"],
                "reward": reward
            })

if __name__ == "__main__":
    main()
