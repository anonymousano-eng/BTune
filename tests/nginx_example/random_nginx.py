import random
from typing import Any, Mapping, List, Union
import subprocess
import time
import csv
import os
import logging
from oppertune.core.values import Categorical, Integer

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ParameterTypes = Union[Categorical, Integer]

class Application:
    def __init__(self):
        self.conf_path = os.path.join(os.path.dirname(__file__), "nginx1.conf")
        self.root_path = os.path.dirname(__file__)
        self.metrics = {"throughput": 0.0, "latency": 1000.0}

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

    client_body_timeout {parameters["client_body_timeout"]};
    client_header_timeout {parameters["client_header_timeout"]};
    send_timeout {parameters["send_timeout"]};
    server_names_hash_bucket_size {parameters["server_names_hash_bucket_size"]};
    client_max_body_size {parameters["client_max_body_size"]}m;
    keepalive_requests {parameters["keepalive_requests"]};
    reset_timedout_connection {parameters["reset_timedout_connection"]};
    server_tokens {parameters["server_tokens"]};
    gzip on;
    gzip_comp_level {parameters["gzip_comp_level"]};
    gzip_min_length {parameters["gzip_min_length"]};
    gzip_buffers {parameters["gzip_buffers"]};
    open_file_cache_valid {parameters["open_file_cache_valid"]};
    open_file_cache_min_uses {parameters["open_file_cache_min_uses"]};

    tcp_nopush {parameters["tcp_nopush"]};
    tcp_nodelay {parameters["tcp_nodelay"]};
    client_header_buffer_size {parameters["client_header_buffer_size"]};
    large_client_header_buffers {parameters["large_client_header_buffers"]};
    client_body_buffer_size {parameters["client_body_buffer_size"]};
    
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
            return

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
            return

        try:
            subprocess.run(
                ["nginx", "-s", "stop", "-c", self.conf_path, "-p", self.root_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
            )
            subprocess.run(["pkill", "-9", "nginx"], check=False)
            time.sleep(1)
        except Exception as e:
            logger.error(f"Failed to stop Nginx: {e}")

        try:
            result = subprocess.run(
                ["nginx", "-c", self.conf_path, "-p", self.root_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
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
                return
            logger.info("Nginx started successfully")
        except Exception as e:
            logger.error(f"Failed to start Nginx: {e}")
            return

        try:
            result = subprocess.run(
                ["wrk", "-t12", "-c200", "-d5s", "http://localhost:8080/"],
                capture_output=True, text=True, check=True
            ).stdout
            logger.debug(f"wrk output:\n{result}")
        except Exception as e:
            logger.error(f"Failed to run wrk test: {e}")
            result = ""

        try:
            subprocess.run(
                ["nginx", "-s", "stop", "-c", self.conf_path, "-p", self.root_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
            )
            subprocess.run(["pkill", "-9", "nginx"], check=False)
            time.sleep(1)
        except Exception as e:
            logger.error(f"Failed to stop Nginx: {e}")

        throughput = 0.0
        latency = 0.0
        for line in result.splitlines():
            if "Requests/sec" in line:
                try:
                    throughput = float(line.split(":")[1].strip())
                except (IndexError, ValueError):
                    throughput = 0.0
            if "Latency" in line and "Thread" not in line:
                latency_str = line.split()[1]
                try:
                    if "us" in latency_str:
                        latency = float(latency_str.replace("us", "")) / 1000
                    elif "ms" in latency_str:
                        latency = float(latency_str.replace("ms", ""))
                    elif "s" in latency_str:
                        latency = float(latency_str.replace("s", "")) * 1000
                except (IndexError, ValueError):
                    latency = 1000.0
        self.metrics = {"throughput": throughput, "latency": latency}

    def get_metrics(self):
        return self.metrics

def calculate_reward(metrics: Mapping[str, Any]) -> float:
    throughput = metrics.get("throughput", 0)
    latency = metrics.get("latency", 1000)
    norm_throughput = min(throughput / 100000, 1.0)
    norm_latency = min(latency / 1000, 1.0)
    reward = (norm_throughput * (1 - norm_latency)) ** 4
    return reward

class RandomSearch:
    def __init__(self, parameters: List[ParameterTypes], random_seed: int = 42):
        self.parameters = parameters
        random.seed(random_seed)
    
    def predict(self) -> Mapping[str, Any]:
        sampled_params = {}
        for param in self.parameters:
            if isinstance(param, Integer):
                sampled_params[param.name] = random.randint(param.min, param.max)
            elif isinstance(param, Categorical):
                sampled_params[param.name] = random.choice(param.categories)
        return sampled_params

def main() -> None:
    app = Application()
    parameters_to_tune = [
        Integer("worker_processes", val=1, min=1, max=8),
        Integer("worker_connections", val=1024, min=128, max=8192),
        Categorical("keepalive", val="on", categories=["on", "off"]),
        Integer("client_body_timeout", val=60, min=10, max=300),
        Integer("client_header_timeout", val=60, min=10, max=300),
        Integer("send_timeout", val=60, min=10, max=300),
        Integer("server_names_hash_bucket_size", val=64, min=32, max=256),
        Integer("client_max_body_size", val=100, min=1, max=150),
        Integer("keepalive_requests", val=50000, min=1000, max=80000),
        Categorical("reset_timedout_connection", val="on", categories=["on", "off"]),
        Categorical("server_tokens", val="off", categories=["on", "off"]),
        Integer("gzip_comp_level", val=3, min=1, max=9),
        Integer("gzip_min_length", val=256, min=128, max=2048),
        Categorical("gzip_buffers", val="16 8k", categories=("4 4k","4 8k","4 16k","32 8k","16 4k","8 8k", "8 16k", "16 8k", "16 16k" ,"2 8k", "2 16k")),
        Integer("open_file_cache_valid", val=30, min=10, max=50),
        Integer("open_file_cache_min_uses", val=2, min=1, max=5),
        Categorical("accept_mutex", val="on", categories=["on", "off"]),
        Categorical("multi_accept", val="on", categories=["on", "off"]),
        Integer("worker_rlimit_nofile", val=102400, min=1, max=655350),
        Categorical("tcp_nopush", val="on", categories=["on", "off"]),
        Categorical("tcp_nodelay", val="on", categories=["on", "off"]),
        Categorical("client_body_buffer_size", val="128k", categories=("16k","32k","64k", "128k","256k","512k", "1024k","2048k")),
        Categorical("client_header_buffer_size", val="4k", categories=("2k","4k", "8k","16k","32k", "64k")),
        Categorical("large_client_header_buffers", val="4 8k", categories=("4 4k","4 8k","4 16k","4 32k","4 64k","4 128k","8 8k", "8 16k", "8 32k", "16 8k", "16 16k" ,"2 8k", "2 16k")),
    ]
    random_search = RandomSearch(parameters_to_tune, random_seed=42)
    with open("random_search_result.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "throughput", "latency", "reward"])
        writer.writeheader()
        for iteration in range(1000):
            try:
                t_algo_predict0 = time.perf_counter()
                predicted_params = random_search.predict()
                t_algo_predict1 = time.perf_counter()
                algo_time_predict_ms = (t_algo_predict1 - t_algo_predict0) * 1000
                print("random search overhead is ", algo_time_predict_ms)
                logger.info(f"[{iteration}] Predicted parameters: {predicted_params}")

                app.set_parameters(predicted_params)
                app.run()
                metrics = app.get_metrics()
                reward = calculate_reward(metrics)
                logger.info(f"[{iteration}] Throughput: {metrics['throughput']:.2f} req/s, Latency: {metrics['latency']:.2f} ms, Reward: {reward:.4f}")

                writer.writerow({
                    "iteration": iteration,
                    "throughput": metrics["throughput"],
                    "latency": metrics["latency"],
                    "reward": reward
                })
                f.flush()
            except KeyboardInterrupt:
                logger.info("User interrupted execution")
                break
            except Exception as e:
                logger.error(f"Iteration {iteration} failed: {e}")
                continue

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Program failed: {e}")
    finally:
        subprocess.run(["pkill", "-9", "nginx"], check=False)
