from typing import Any, Mapping, List, Union
from oppertune.algorithms.hybrid_solver import HybridSolver
from oppertune.core.values import Integer, Real, Categorical
from oppertune.algorithms.moe_ppo_tuner import BottleneckIdentifier
import subprocess
import time
import csv
import os
import threading
import logging
import traceback
import psutil
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TOTAL_REQS   = 1000_000 
THREADS      = 8 
CONCURRENCY  = 20
RATIO        = "1:1"
ITERATIONS   = 200          

ParameterTypes = Union[Integer, Real, Categorical]

class Application:
    def __init__(self):
        self.memcached_proc = None
        self.bottleneck_identifier = None
        self.metrics = {"throughput": 0.0, "latency": 1000.0}
        self.error_log = "memcached_error.log"

    def set_parameters(self, parameters: Mapping[str, Any]):
        self.threads = parameters["threads"]
        self.conn_limit = parameters["conn_limit"]
        self.memory = parameters["memory"]
        self.max_reqs_per_event = parameters["max_reqs_per_event"]
        self.slab_growth_factor = parameters["slab_growth_factor"]
        self.lock_memory = parameters["lock_memory"]
        self.large_pages = parameters["large_pages"]
        self.verbose = parameters["verbose"]
        self.listen_backlog = parameters["listen_backlog"]

    def run(self):
        try:
            subprocess.run(["pkill", "-9", "memcached"], check=False)
            time.sleep(1)
        except subprocess.SubprocessError as e:
            logger.error(f"kill memcached failed: {e}")

        try:
            result = subprocess.run(["netstat", "-tuln"], capture_output=True, text=True)
            if ":11211" in result.stdout:
                logger.warning("port 11211 being use，try to rlease")
                subprocess.run(["fuser", "-k", "11211/tcp"], check=False)
                time.sleep(1)
        except subprocess.SubprocessError as e:
            logger.warning(f"check port failed: {e}")

        cmd = [
            "memcached",
            "-u", "nobody",
            "-p", "11211",
            "-t", str(self.threads),
            "-c", str(self.conn_limit),
            "-m", str(self.memory),
            "-R", str(self.max_reqs_per_event),
            "-f", str(self.slab_growth_factor),
            "-b", str(self.listen_backlog),
            "-v" * self.verbose
        ]
        if self.lock_memory == "on":
            cmd.append("-k")
        if self.large_pages == "on":
            cmd.append("-L")
        
        try:
            with open(self.error_log, "a") as f:
                self.memcached_proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=f
                )
            time.sleep(2)
            if self.memcached_proc.poll() is not None:
                logger.error("Memcached launch failed")
                with open(self.error_log, "r") as f:
                    logger.error(f"Memcached 错误日志: {f.read()}")
                return [0.0, 0.0, 0.0, 0.0]
            logger.info("Memcached launch done")
        except subprocess.SubprocessError as e:
            logger.error(f"launch memcached failed: {e}")
            return [0.0, 0.0, 0.0, 0.0]
            
        try:
            cmd_init = [
                "memtier_benchmark", "-s", "localhost", "-p", "11211",
                "--initial-load=1000"
            ]
            subprocess.run(cmd_init, capture_output=True, text=True, check=True)
            logger.debug("Memcached init done")
        except subprocess.SubprocessError as e:
            logger.error(f"init Memcached failed: {e}")
            self._stop_memcached()
            return [0.0, 0.0, 0.0, 0.0]

        util_measurements = []
        def monitor_resources():
            start_time = time.time()
            while time.time() - start_time < 10:
                if self.bottleneck_identifier:
                    util = self.bottleneck_identifier.get_system_util()
                    util_measurements.append(util)
                time.sleep(0.2)

        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()

        try:
            cmd = [
                "memtier_benchmark", "-s", "localhost", "-p", "11211",
                "-t", str(THREADS), "-c", str(CONCURRENCY), "-n", str(TOTAL_REQS),
                "--ratio", RATIO
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            ).stdout
            logger.debug(f"memtier_benchmark output:\n{result}")
        except subprocess.SubprocessError as e:
            logger.error(f"memtier_benchmark failed: {e}\output: {getattr(e, 'output', '')}")
            result = ""
        finally:
            monitor_thread.join()

        bottleneck_vector = [0.0, 0.0, 0.0, 0.0]
        if util_measurements:
            util_array = np.array(util_measurements)
            avg_util = np.mean(util_array, axis=0)
            thresholds = [
                self.bottleneck_identifier.cpu_thres,
                self.bottleneck_identifier.mem_thres,
                self.bottleneck_identifier.io_thres,
                self.bottleneck_identifier.net_thres
            ]
            bottleneck_vector = [1.0 if util >= thresh else 0.0 for util, thresh in zip(avg_util, thresholds)]
            logger.info(f"average util: CPU={avg_util[0]:.2f}, MEM={avg_util[1]:.2f}, IO={avg_util[2]:.2f}, NET={avg_util[3]:.2f}")
            logger.info(f"bottleneck vector: {bottleneck_vector} (threshold: CPU={thresholds[0]}, MEM={thresholds[1]}, IO={thresholds[2]}, NET={thresholds[3]})")
        else:
            logger.warning("util_measurements empty")

        # 更新 util_history
        if self.bottleneck_identifier and util_measurements:
            self.bottleneck_identifier.util_history.clear()
            for util in util_measurements:
                self.bottleneck_identifier.util_history.append(util)
            logger.info(f"collected {len(util_measurements)} times utils")

        self._stop_memcached()

        throughput, latency = 0.0, 1000.0
        for line in result.strip().splitlines():
            if "Totals" in line:
                try:
                    fields = line.split()
                    if len(fields) >= 5:
                        throughput = float(fields[1])  # Ops/sec
                        latency = float(fields[4])     # Avg. Latency (ms)
                        logger.info(f"parse Totals: throughput={throughput:.2f} ops/s, latency={latency:.2f} ms")
                        break
                except (ValueError, IndexError) as e:
                    logger.warning(f"parse Totals failed: {line} | error: {e}")
                    continue

        if throughput == 0.0:
            self.metrics = {"throughput": 0.0, "latency": 1000.0}
        else:
            self.metrics = {"throughput": throughput, "latency": latency}

        return bottleneck_vector

    def _stop_memcached(self):
        try:
            if self.memcached_proc and self.memcached_proc.poll() is None:
                self.memcached_proc.terminate()
                self.memcached_proc.wait(timeout=5)
            subprocess.run(["pkill", "-9", "memcached"], check=False)
            time.sleep(1)
            logger.debug("Memcached terminated")
        except subprocess.SubprocessError as e:
            logger.error(f"kill memcached failed: {e}")

    def get_metrics(self):
        return self.metrics

    def __del__(self):
        self._stop_memcached()

def calculate_reward(metrics: Mapping[str, Any]) -> float:

    throughput = metrics.get("throughput", 0)
    latency = metrics.get("latency", 1000)
    t_normalized = throughput / 100000.0
    return t_normalized / (latency + 1)

def main() -> None:
    app = Application()
    bottleneck_identifier = BottleneckIdentifier(cpu_thres=0.3, mem_thres=0.2, io_thres=0.1, net_thres=0.1)
    app.bottleneck_identifier = bottleneck_identifier

    parameters_to_tune = [

        {'param': Integer("threads", val=4, min=1, max=16), 'expert_type': 'CPU'},
        {'param': Integer("verbose", val=1, min=0, max=3), 'expert_type': 'CPU'},

        {'param': Integer("memory", val=64, min=64, max=4096), 'expert_type': 'MEM'},
        {'param': Real("slab_growth_factor", val=1.25, min=1.0, max=2.0), 'expert_type': 'MEM'},
        {'param': Categorical("large_pages", val="off", categories=["on", "off"]), 'expert_type': 'MEM'},
        {'param': Categorical("lock_memory", val="off", categories=["on", "off"]), 'expert_type': 'MEM'},

        {'param': Integer("conn_limit", val=1024, min=512, max=4096), 'expert_type': 'NET'},
        {'param': Integer("listen_backlog", val=1024, min=10, max=5000), 'expert_type': 'NET'},
        {'param': Integer("max_reqs_per_event", val=20, min=10, max=100), 'expert_type': 'NET'},
    ]

    tuning_instance = HybridSolver(
        parameters=[p['param'] for p in parameters_to_tune],
        categorical_algorithm="exponential_weights_slates",
        categorical_algorithm_args={"random_seed": 42},
        numerical_algorithm="bluefin",
        numerical_algorithm_args={"feedback": 2, "eta": 0.01, "delta": 0.1},
    )

    with open("hybrid_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "iteration", "throughput", "latency", "reward",
            "cpu_bottleneck", "mem_bottleneck", "io_bottleneck", "net_bottleneck"
        ])
        writer.writeheader()

        for iteration in range(ITERATIONS):
            try:
                predicted_params, request_id = tuning_instance.predict()
                logger.info(f"[{iteration}] predicted param: {predicted_params}")
                app.set_parameters(predicted_params)
                bottleneck_vector = app.run()
                metrics = app.get_metrics()
                reward = calculate_reward(metrics)
                logger.info(f"[{iteration}] reward: {reward:.4f}, metrics: {metrics}")
                tuning_instance.store_reward(request_id, reward)
                tuning_instance.set_reward(reward)

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
                logger.info("program interrupt")
                break
            except Exception as e:
                logger.error(f"iter {iteration} failed: {str(e)}")
                logger.error(traceback.format_exc())
                continue

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("program interrupt")
    except Exception as e:
        logger.error(f"program error: {e}")
        logger.error(traceback.format_exc())
    finally:
        subprocess.run(["pkill", "-9", "memcached"], check=False)