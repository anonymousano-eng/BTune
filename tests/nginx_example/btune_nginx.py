from typing import Any, Mapping, List, Union
from oppertune.algorithms.moe_ppo_tuner import MoEPPOAlgorithm, BottleneckIdentifier
from oppertune.core.values import Integer, Categorical
import subprocess
import time
import csv
import os
import threading
import logging
import traceback
import psutil
import numpy as np

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ParameterTypes = Union[Integer, Categorical]

class Application:
    def __init__(self):
        self.nginx_proc = None
        self.conf_path = os.path.join(os.path.dirname(__file__), "nginx.conf")
        self.root_path = os.path.dirname(__file__)
        self.metrics = {"throughput": 0.0, "latency": 1000.0}
        self.bottleneck_identifier = None

    def set_parameters(self, parameters: Mapping[str, Any]):
        keepalive_timeout = "65" if parameters["keepalive"] == "on" else "0"
        mime_types = """
types {
    text/html                             html htm shtml;
    text/css                              css;
    text/xml                              xml;
    image/gif                             gif;
    image/jpeg                            jpeg jpg;
    application/javascript                js;
    application/atom+xml                  atom;
    application/rss+xml                   rss;
    text/plain                            txt;
    application/octet-stream              bin exe dll;
}
"""
        with open(self.conf_path, "w") as f:
            f.write(f'''
worker_processes {parameters["worker_processes"]};
worker_rlimit_nofile {parameters["worker_rlimit_nofile"]};
error_log {os.path.join(self.root_path, "var/log/nginx/error.log")} crit;

events {{
    worker_connections {parameters["worker_connections"]};
    accept_mutex {parameters["accept_mutex"]};
    multi_accept {parameters["multi_accept"]};
}}

http {{
    {mime_types}
    default_type  application/octet-stream;
    access_log off;

    sendfile        on;
    keepalive_timeout {keepalive_timeout};

    client_body_timeout 60;
    client_header_timeout 60;
    send_timeout 60;
    server_names_hash_bucket_size 64;
    client_max_body_size 100m;
    keepalive_requests 50000;
    reset_timedout_connection off;
    server_tokens off;
    gzip on;
    gzip_comp_level {parameters["gzip_comp_level"]};
    gzip_min_length {parameters["gzip_min_length"]};
    gzip_buffers 16 8k;
    open_file_cache_valid 30;
    open_file_cache_min_uses 2;

    tcp_nopush off;
    tcp_nodelay {parameters["tcp_nodelay"]};
    client_header_buffer_size 4k;
    large_client_header_buffers 4 8k;
    client_body_buffer_size 128k;
    
    proxy_temp_path {os.path.join(self.root_path, "var/tmp/nginx/proxy")};
    client_body_temp_path {os.path.join(self.root_path, "var/tmp/nginx/client")};
    server {{
        listen 8080;
        location / {{
            root {os.path.join(self.root_path, "html")};
            index index.html;
        }}
    }}
}}
''')

    def run(self):
        try:
            os.makedirs(os.path.join(self.root_path, "var/log/nginx"), exist_ok=True)
            os.makedirs(os.path.join(self.root_path, "var/run/nginx"), exist_ok=True)
            os.makedirs(os.path.join(self.root_path, "var/tmp/nginx/client"), exist_ok=True)
            os.makedirs(os.path.join(self.root_path, "var/tmp/nginx/proxy"), exist_ok=True)
            os.makedirs(os.path.join(self.root_path, "html"), exist_ok=True)
            index_path = os.path.join(self.root_path, "html/index.html")
            if not os.path.exists(index_path):
                with open(index_path, "wb") as f:
                    f.write(b"X" * 1024 * 1024)
                logger.info(f"Created {index_path}")
            os.chmod(self.root_path, 0o755)
            os.chmod(os.path.join(self.root_path, "var"), 0o755)
            os.chmod(os.path.join(self.root_path, "var/log"), 0o755)
            os.chmod(os.path.join(self.root_path, "var/log/nginx"), 0o755)
            os.chmod(os.path.join(self.root_path, "var/tmp"), 0o755)
            os.chmod(os.path.join(self.root_path, "var/tmp/nginx"), 0o755)
        except Exception as e:
            logger.error(f"Failed to create directories or files: {e}")
            return [0.0, 0.0, 0.0, 0.0]

        try:
            result = subprocess.run(["netstat", "-tuln"], capture_output=True, text=True)
            if ":8080" in result.stdout:
                logger.warning("Port 8080 is in use, attempting to clean")
                subprocess.run(["fuser", "-k", "8080/tcp"], check=False)
                time.sleep(1)
        except Exception as e:
            logger.warning(f"Failed to check port: {e}")

        try:
            result = subprocess.run(
                ["nginx", "-t", "-c", self.conf_path, "-p", self.root_path],
                capture_output=True, text=True, check=True
            )
            logger.debug(f"Nginx config test: {result.stderr}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Nginx config test failed: {e.stderr}")
            return [0.0, 0.0, 0.0, 0.0]

        self._stop_nginx()
        if not self._start_nginx():
            return [0.0, 0.0, 0.0, 0.0]

        util_measurements = []
        def monitor_resources():
            start_time = time.time()
            while time.time() - start_time < 5:
                if self.bottleneck_identifier:
                    util = self.bottleneck_identifier.get_system_util()
                    util_measurements.append(util)
                time.sleep(0.2)

        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()

        result = self._run_wrk_test()
        monitor_thread.join()

        bottleneck_vector = [0.0, 0.0, 0.0, 0.0]
        if util_measurements:
            util_array = np.array(util_measurements)
            avg_util = np.mean(util_array, axis=0)
            thresholds = [0.3, 0.2, 0.1, 0.1]
            bottleneck_vector = [1.0 if util >= thresh else 0.0 for util, thresh in zip(avg_util, thresholds)]
            logger.info(f"Runtime average utilization: CPU={avg_util[0]:.2f}, MEM={avg_util[1]:.2f}, IO={avg_util[2]:.2f}, NET={avg_util[3]:.2f}")
            logger.info(f"Runtime bottleneck vector: {bottleneck_vector}")

        if self.bottleneck_identifier and util_measurements:
            self.bottleneck_identifier.util_history.clear()
            for util in util_measurements:
                self.bottleneck_identifier.util_history.append(util)
            logger.info(f"Collected {len(util_measurements)} resource utilization measurements")

        self._stop_nginx()
        self._parse_metrics(result)

        try:
            cpu_util = psutil.cpu_percent(interval=0.1) / 100.0
            mem_util = psutil.virtual_memory().percent / 100.0
            disk_io = psutil.disk_io_counters()
            io_util = (disk_io.read_bytes + disk_io.write_bytes) / 1024**2 / 5
            net_io = psutil.net_io_counters()
            net_util = (net_io.bytes_sent + net_io.bytes_recv) / 1024**2 / 5
            logger.info(f"Final resource utilization: CPU={cpu_util:.2f}, MEM={mem_util:.2f}, IO={io_util:.2f} MB/s, NET={net_util:.2f} MB/s")
        except Exception as e:
            logger.warning(f"Failed to get final resource utilization: {e}")

        return bottleneck_vector

    def _start_nginx(self):
        try:
            result = subprocess.run(
                ["nginx", "-c", self.conf_path, "-p", self.root_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            time.sleep(2)
            check = subprocess.run(
                ["ps", "aux"], capture_output=True, text=True
            )
            if "nginx: master process" not in check.stdout:
                logger.error("Nginx failed to start, no master process found")
                error_log = os.path.join(self.root_path, "var/log/nginx/error.log")
                if os.path.exists(error_log):
                    with open(error_log, "r") as f:
                        logger.error(f"Nginx error log: {f.read()}")
                return False
            logger.info("Nginx started successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to start Nginx: {e}")
            return False

    def _stop_nginx(self):
        try:
            subprocess.run(
                ["nginx", "-s", "stop", "-c", self.conf_path, "-p", self.root_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False
            )
            subprocess.run(["pkill", "-9", "nginx"], check=False)
            time.sleep(1)
        except Exception as e:
            logger.error(f"Failed to stop Nginx: {e}")

    def _run_wrk_test(self):
        try:
            result = subprocess.run(
                ["wrk", "-t12", "-c200", "-d5s", "http://localhost:8080/"],
                capture_output=True, 
                text=True,
                check=True
            ).stdout
            logger.debug(f"wrk output:\n{result}")
            return result
        except Exception as e:
            logger.error(f"Failed to run wrk test: {e}")
            return ""

    def _parse_metrics(self, result: str):
        for line in result.splitlines():
            if "Requests/sec" in line:
                try:
                    self.metrics["throughput"] = float(line.split(":")[1].strip())
                except (IndexError, ValueError):
                    self.metrics["throughput"] = 0.0
            if "Latency" in line and "Thread" not in line:
                latency_str = line.split()[1]
                try:
                    if "us" in latency_str:
                        self.metrics["latency"] = float(latency_str.replace("us", "")) / 1000
                    elif "ms" in latency_str:
                        self.metrics["latency"] = float(latency_str.replace("ms", ""))
                    elif "s" in latency_str:
                        self.metrics["latency"] = float(latency_str.replace("s", "")) * 1000
                except (IndexError, ValueError):
                    self.metrics["latency"] = 1000.0

    def get_metrics(self):
        return self.metrics

    def __del__(self):
        self._stop_nginx()

def calculate_reward(metrics: Mapping[str, Any]) -> float:
    throughput = metrics.get("throughput", 0)
    latency = metrics.get("latency", 1000)
    norm_throughput = min(throughput / 100000, 1.0)
    norm_latency = min(latency / 1000, 1.0)
    reward = (norm_throughput * (1 - norm_latency)) ** 4
    return reward

def main() -> None:
    app = Application()
    bottleneck_identifier = BottleneckIdentifier(cpu_thres=0.3, mem_thres=0.2, io_thres=0.1, net_thres=0.1)
    app.bottleneck_identifier = bottleneck_identifier
    
    parameters_to_tune: list[dict[str, Any]] = [
        {'param': Integer("worker_processes",         1,     1,     8),             'expert_type': 'CPU'},
        {'param': Integer("worker_connections",       1024,  128,   8192),          'expert_type': 'CPU'},
        {'param': Categorical("accept_mutex",         "on",  ["on", "off"]),        'expert_type': 'CPU'},
        {'param': Categorical("multi_accept",         "on",  ["on", "off"]),        'expert_type': 'CPU'},
        {'param': Integer("worker_rlimit_nofile",     102400,1,     655350),        'expert_type': 'CPU'},
        {'param': Integer("gzip_min_length",          256,   128,   2048),          'expert_type': 'MEM'},
        {'param': Categorical("gzip_buffers",         "16 8k",
                              ["4 4k","4 8k","4 16k","8 8k","8 16k",
                               "16 4k","16 8k","16 16k","32 8k","2 8k","2 16k"]),
         'expert_type': 'MEM'},
        {'param': Categorical("client_body_buffer_size",  "128k",
                              ["16k","32k","64k","128k","256k","512k","1024k","2048k"]),
         'expert_type': 'MEM'},
        {'param': Categorical("client_header_buffer_size","4k",
                              ["2k","4k","8k","16k","32k","64k"]),
         'expert_type': 'MEM'},
        {'param': Categorical("large_client_header_buffers", "4 8k",
                              ["4 4k","4 8k","4 16k","4 32k","4 64k","4 128k",
                               "8 8k","8 16k","8 32k","16 8k","16 16k","2 8k","2 16k"]),
         'expert_type': 'MEM'},
        {'param': Integer("gzip_comp_level",          3,     1,     9),             'expert_type': 'IO'},
        {'param': Integer("open_file_cache_valid",    30,    10,    50),            'expert_type': 'IO'},
        {'param': Integer("open_file_cache_min_uses", 2,     1,     5),             'expert_type': 'IO'},
        {'param': Integer("send_timeout",             60,    10,    300),           'expert_type': 'IO'},
        {'param': Integer("client_body_timeout",      60,    10,    300),           'expert_type': 'IO'},
        {'param': Integer("client_header_timeout",    60,    10,    300),           'expert_type': 'IO'},
        {'param': Categorical("keepalive",            "on",  ["on", "off"]),        'expert_type': 'NET'},
        {'param': Integer("keepalive_requests",       50000, 1000,  80000),         'expert_type': 'NET'},
        {'param': Categorical("tcp_nopush",           "on",  ["on", "off"]),        'expert_type': 'NET'},
        {'param': Categorical("tcp_nodelay",          "on",  ["on", "off"]),        'expert_type': 'NET'},
        {'param': Integer("server_names_hash_bucket_size", 64, 32, 256),            'expert_type': 'NET'},
        {'param': Integer("client_max_body_size",     100,   1,     150),           'expert_type': 'NET'},
        {'param': Categorical("reset_timedout_connection", "on", ["on", "off"]),    'expert_type': 'NET'},
        {'param': Categorical("server_tokens",        "off", ["on", "off"]),        'expert_type': 'NET'},
    ]

    tuning_instance = MoEPPOAlgorithm(parameters_to_tune)
    tuning_instance.bottleneck_identifier = bottleneck_identifier

    with open("btune_nginx_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "iteration", "throughput", "latency", "reward",
            "cpu_bottleneck", "mem_bottleneck", "io_bottleneck", "net_bottleneck"
        ])
        writer.writeheader()

        for iteration in range(1000):
            try:
                aaa = time.perf_counter()
                predicted_params = tuning_instance.predict()
                bbb = time.perf_counter()
                state_overhead = (bbb - aaa) * 1000
                print("moe_ppo overhead is ", state_overhead)
                logger.info(f"[{iteration}] Predicted parameters: {predicted_params}")

                app.set_parameters(predicted_params)
                bottleneck_vector = app.run()
                if bottleneck_vector is None:
                    bottleneck_vector = [0.0, 0.0, 0.0, 0.0]
                
                metrics = app.get_metrics()
                reward = calculate_reward(metrics)
                logger.info(f"[{iteration}] Throughput: {metrics['throughput']:.2f} req/s, Latency: {metrics['latency']:.2f} ms, Reward: {reward:.4f}")
                logger.info(f"[{iteration}] Runtime bottleneck vector: {bottleneck_vector}")

                tuning_instance.set_reward(reward, metrics)

                writer.writerow({
                    "iteration": iteration,
                    "throughput": metrics["throughput"],
                    "latency": metrics["latency"],
                    "reward": reward,
                    "cpu_bottleneck": bottleneck_vector[0],
                    "mem_bottleneck": bottleneck_vector[1],
                    "io_bottleneck": bottleneck_vector[2],
                    "net_bottleneck": bottleneck_vector[3]
                })
                f.flush()

            except KeyboardInterrupt:
                logger.info("User interrupted execution")
                break
            except Exception as e:
                logger.error(f"Iteration {iteration} failed: {str(e)}")
                logger.error(traceback.format_exc())
                continue

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Program failed: {e}")
        logger.error(traceback.format_exc())
    finally:
        subprocess.run(["pkill", "-9", "nginx"], check=False)
