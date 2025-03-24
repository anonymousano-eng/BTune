from __future__ import annotations

import csv
import json
import logging
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping

import psutil

from oppertune.algorithms.hybrid_solver import HybridSolver
from oppertune.core.values import Integer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger(__name__)


ROOT = Path(__file__).resolve().parent
RUNTIME_SEC = 30
ITERATIONS = 1000
CPU_MAX_LOAD = 100
TEST_FILE = "benchfile"
RESULT_CSV = ROOT / "opper_mix_results.csv"



PARAMETERS = [
    Integer("cpu_workers",   val=1,  min=1, max=psutil.cpu_count(logical=False) or 8),
    Integer("cpu_load",      val=50, min=20, max=CPU_MAX_LOAD),
    Integer("iodepth",       val=4,  min=1, max=64),
    Integer("blocksize",     val=8,  min=4, max=128),
    Integer("rwmix",         val=50, min=0, max=100), 
]

class MixedWorkload:

    def __init__(self) -> None:
        self.metrics: Dict[str, float] = {"throughput": 0.0, "latency": 1000.0}

    @staticmethod
    def _mean_lat_ms(job: Mapping[str, Any], rw: str) -> float:
        for key in ("clat_ns", "clat", "lat_ns", "lat"):
            if key in job[rw]:
                mean = job[rw][key]["mean"]
                return mean / (1e6 if key.endswith("_ns") else 1e3)
        return 1000.0
    # ------------------------------------

    def run_once(self, params: Mapping[str, Any]) -> None:
        cpu_workers = params["cpu_workers"]
        cpu_load    = params["cpu_load"]
        iodepth     = params["iodepth"]
        blksz_kb    = params["blocksize"]
        rwmix       = params["rwmix"]

        # 1) stress‑ng
        stress_cmd = [
            "stress-ng", "--cpu", str(cpu_workers),
            "--cpu-load", str(cpu_load),
            "--timeout", f"{RUNTIME_SEC}s",
        ]
        stress_p = subprocess.Popen(stress_cmd,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
        time.sleep(1)

        # 2) fio
        fio_cmd = [
            "fio", "--output-format=json",
            "--name=mix",
            "--rw=randrw", f"--rwmixread={rwmix}",
            f"--bs={blksz_kb}k",
            f"--iodepth={iodepth}",
            "--ioengine=psync",
            "--direct=1",
            "--size=1G",
            f"--runtime={RUNTIME_SEC}", "--time_based",
            f"--filename={TEST_FILE}",
            "--group_reporting",
        ]

        bw_mb = 0.0
        lat_ms = 1000.0
        try:
            fio_out = subprocess.check_output(fio_cmd, text=True)
            job = json.loads(fio_out)["jobs"][0]
            bw_mb = (job["read"]["bw_bytes"] + job["write"]["bw_bytes"]) / 1048576.0
            lat_ms = max(
                self._mean_lat_ms(job, "read")  if job["read"]["io_bytes"] else 0.0,
                self._mean_lat_ms(job, "write") if job["write"]["io_bytes"] else 0.0,
            )
        except Exception as exc:  # broad catch OK here for robustness
            log.error(f"fio run/parse error: {exc}")
            log.debug(traceback.format_exc())
        finally:
            stress_p.terminate()
            try:
                stress_p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                stress_p.kill()

        self.metrics = {"throughput": bw_mb, "latency": lat_ms}

    def reward(self) -> float:
        thr_norm = min(self.metrics["throughput"] / 500.0, 1.0)
        lat_norm = min(self.metrics["latency"] / 100.0, 1.0)
        return thr_norm * (1.0 - lat_norm)



def main() -> None:
    workload = MixedWorkload()

    solver = HybridSolver(
        PARAMETERS,
        categorical_algorithm="exponential_weights_slates",
        categorical_algorithm_args={"random_seed": 42},
        numerical_algorithm="bluefin",
        numerical_algorithm_args={"feedback": 2, "eta": 0.01, "delta": 0.1, "random_seed": 42},
    )

    with RESULT_CSV.open("w", newline="") as fp:
        fieldnames = [
            "iter", "throughput", "latency", "reward",
            *[p.name for p in PARAMETERS],
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(ITERATIONS):
            try:
                params, req_id = solver.predict()  # type: ignore[attr-defined]
                log.info(f"[{i:3d}] params = {params}")

                workload.run_once(params)
                R = workload.reward()

                m = workload.metrics
                log.info(
                    f"[{i:3d}] BW={m['throughput']:.1f} MB/s | "
                    f"Lat={m['latency']:.2f} ms | R={R:.4f}"
                )

                solver.store_reward(req_id, R)     # type: ignore[attr-defined]
                solver.set_reward(R)               # type: ignore[attr-defined]

                writer.writerow({
                    "iter": i,
                    "throughput": m["throughput"],
                    "latency": m["latency"],
                    "reward": R,
                    **params,
                })
                fp.flush()

            except KeyboardInterrupt:
                log.info("User interrupted, exiting...")
                break
            except Exception as exc:
                log.error(f"Iteration {i} failed: {exc}")
                log.debug(traceback.format_exc())
                continue


if __name__ == "__main__":
    try:
        main()
    finally:
        subprocess.run(["pkill", "-9", "stress-ng", "fio"], check=False)
