from __future__ import annotations

import csv
import json
import logging
import random
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping

import psutil


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger(__name__)
# -------------------------------------------


ROOT = Path(__file__).resolve().parent
RUNTIME_SEC = 30 
ITERATIONS = 1000  
CPU_MAX_LOAD = 100  
TEST_FILE = "benchfile" 
RESULT_CSV = ROOT / "random_mix_results.csv"
# =============================================


PARAM_SPACE: List[Dict[str, Any]] = [
    # int 型：low ≤ value ≤ high
    {"name": "cpu_workers", "type": "int", "low": 1, "high": psutil.cpu_count(logical=False) or 8},
    {"name": "cpu_load",    "type": "int", "low": 20, "high": CPU_MAX_LOAD},
    {"name": "iodepth",     "type": "int", "low": 1, "high": 64},
    {"name": "blocksize",   "type": "int", "low": 4, "high": 128},
    {"name": "rwmix",       "type": "int", "low": 0, "high": 100},
]
# ============================================


class SimpleRandomSearch:

    def __init__(self, space: List[Dict[str, Any]]):
        self.space = space
        self._req_id = 0

    def _sample(self, spec: Dict[str, Any]):
        if spec["type"] == "int":
            return random.randint(spec["low"], spec["high"])
        raise ValueError(f"Unsupported type: {spec}")

    def predict(self):
        cfg = {spec["name"]: self._sample(spec) for spec in self.space}
        rid = self._req_id
        self._req_id += 1
        return cfg, rid

    def store_reward(self, rid: int, reward: float):
        log.debug(f"store_reward(id={rid}, r={reward:.4f})")

    def set_reward(self, reward: float):
        log.debug(f"set_reward(r={reward:.4f})")

# ----------------------------------------------

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

    # ----------------------------------

    def run_once(self, p: Mapping[str, Any]) -> None:
        cpu_workers = p["cpu_workers"]
        cpu_load    = p["cpu_load"]
        iodepth     = p["iodepth"]
        blksz_kb    = p["blocksize"]
        rwmix       = p["rwmix"]

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

        bw_mb, lat_ms = 0.0, 1000.0
        try:
            fio_out = subprocess.check_output(fio_cmd, text=True)
            job = json.loads(fio_out)["jobs"][0]
            bw_mb = (job["read"]["bw_bytes"] + job["write"]["bw_bytes"]) / 1048576.0
            lat_ms = max(
                self._mean_lat_ms(job, "read")  if job["read"]["io_bytes"] else 0.0,
                self._mean_lat_ms(job, "write") if job["write"]["io_bytes"] else 0.0,
            )
        except Exception as exc:
            log.error(f"fio run/parse error: {exc}")
            log.debug(traceback.format_exc())
        finally:
            stress_p.terminate()
            try:
                stress_p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                stress_p.kill()

        self.metrics = {"throughput": bw_mb, "latency": lat_ms}

    # ---------------- Reward ----------------
    def reward(self) -> float:
        thr_norm = min(self.metrics["throughput"] / 500.0, 1.0)
        lat_norm = min(self.metrics["latency"] / 100.0, 1.0)
        return thr_norm * (1.0 - lat_norm)



def main() -> None:
    workload = MixedWorkload()
    solver = SimpleRandomSearch(PARAM_SPACE)

    with RESULT_CSV.open("w", newline="") as fp:
        fieldnames = [
            "iter", "throughput", "latency", "reward",
            *[spec["name"] for spec in PARAM_SPACE],
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(ITERATIONS):
            try:
                params, req_id = solver.predict()
                log.info(f"[{i:3d}] params = {params}")

                workload.run_once(params)
                R = workload.reward()

                m = workload.metrics
                log.info(
                    f"[{i:3d}] BW={m['throughput']:.1f} MB/s | "
                    f"Lat={m['latency']:.2f} ms | R={R:.4f}"
                )

                solver.store_reward(req_id, R)
                solver.set_reward(R)

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
    main()
