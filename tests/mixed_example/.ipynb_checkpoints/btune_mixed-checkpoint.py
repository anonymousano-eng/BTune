from __future__ import annotations
from typing import Any, Mapping, Union
import csv, json, logging, os, signal, subprocess, threading, time, traceback
from collections import deque

import numpy as np
import psutil

from oppertune.algorithms.moe_ppo_tuner import MoEPPOAlgorithm, BottleneckIdentifier
from oppertune.core.values import Integer, Categorical


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s| %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)


RUNTIME   = 30                 
TEST_FILE = "benchfile"         
ITERS     = 1000                
CPU_LOAD_MAX = 100

# ------------------------------------------------------------
class MixedWorkload:
    """stress-ng + fio"""

    def __init__(self):
        self.metrics: dict[str, float] = {"throughput": 0.0, "latency": 1000.0}
        self.btl: BottleneckIdentifier | None = None

    # ---------------------------- #
    def run(self, p: Mapping[str, Any]) -> list[float]:


        cpu_w  = p["cpu_workers"]
        cpu_ld = p["cpu_load"]
        depth  = p["iodepth"]
        bsz    = p["blocksize"]
        mix    = p["rwmix"]
        engine = p["ioengine"]

        # ------------ stress-ng ------------
        stress_p = subprocess.Popen(
            ["stress-ng", "--cpu", str(cpu_w),
             "--cpu-load", str(cpu_ld),
             "--timeout", f"{RUNTIME}s"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1)
        if stress_p.poll() is not None:
            log.error("stress-ng start failed")
            return self._fail(stress_p)

        # ------------ fio ------------
        fio_cmd = [
            "fio", "--output-format=json", "--name=mix",
            "--rw=randrw", f"--rwmixread={mix}",
            f"--bs={bsz}k", f"--iodepth={depth}",
            f"--ioengine={engine}",
            "--direct=1", "--size=1G",
            f"--runtime={RUNTIME}", "--time_based",
            f"--filename={TEST_FILE}", "--group_reporting"
        ]

        util_hist: deque[tuple[float, float, float, float]] = deque(maxlen=150)


        def monitor():
            last_d = psutil.disk_io_counters();  last_n = psutil.net_io_counters(); last_t = time.time()
            end_t = last_t + RUNTIME
            while time.time() < end_t:
                time.sleep(0.2)
                now = time.time();  dt = now - last_t or 1e-6
                cpu = psutil.cpu_percent(None) / 100
                mem = psutil.virtual_memory().percent / 100

                cur_d = psutil.disk_io_counters()
                io_MB = ((cur_d.read_bytes  - last_d.read_bytes) +
                         (cur_d.write_bytes - last_d.write_bytes)) / 1048576 / dt
                last_d = cur_d

                cur_n = psutil.net_io_counters()
                net_MB = ((cur_n.bytes_sent - last_n.bytes_sent) +
                          (cur_n.bytes_recv - last_n.bytes_recv)) / 1048576 / dt
                last_n = cur_n

                util_hist.append((cpu, mem, io_MB, net_MB))
                last_t = now

        mon = threading.Thread(target=monitor, daemon=True);  mon.start()

        try:
            fio_out = subprocess.check_output(fio_cmd, text=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            log.error(f"fio failed:\n{e.output.strip()}")
            return self._fail(stress_p)
        finally:
            mon.join()
            stress_p.wait(5)

        # ------------ parse fio JSON ------------
        try:
            data = json.loads(fio_out)
            job  = data["jobs"][0]
            bw_MB = (job["read"]["bw_bytes"] + job["write"]["bw_bytes"]) / 1048576.0

            def mean_lat(_rw: str) -> float:
                jd = job[_rw]
                for k in ("clat_ns", "lat_ns", "clat", "lat"):
                    if k in jd:
                        mean = jd[k]["mean"]
                        return mean / (1e6 if k.endswith("_ns") else 1e3)
                return 0.0

            lat_ms = max(mean_lat("read"), mean_lat("write"))
        except Exception as e:
            log.error(f"parse fio JSON filed: {e}")
            return self._fail()

        self.metrics = {"throughput": bw_MB, "latency": lat_ms}


        if self.btl and util_hist:
            util_arr = np.array(util_hist)
            cpu_u, mem_u, io_u, net_u = util_arr.mean(axis=0)
            th = [self.btl.cpu_thres, self.btl.mem_thres,
                  self.btl.io_thres,  self.btl.net_thres]
            bvec = [1.0 if u >= t else 0.0
                    for u, t in zip((cpu_u, mem_u, io_u, net_u), th)]
            log.info(f"average util CPU={cpu_u:.2f} MEM={mem_u:.2f} "
                     f"IO={io_u:.2f}MB/s NET={net_u:.2f}MB/s")
            log.info(f"bottleneck vector {bvec}")
            return bvec
        return [0, 0, 0, 0]

    # ---------------------------- #
    def _fail(self, stress_p: subprocess.Popen | None = None) -> list[float]:
        if stress_p: stress_p.kill()
        self.metrics = {"throughput": 0.0, "latency": 1000.0}
        return [0, 0, 0, 0]

    def get_metrics(self): return self.metrics


# ============================================================
def reward(m: Mapping[str, Any]) -> float:
    tp = m["throughput"];  lat = m["latency"]
    return (min(tp/500, 1) * (1 - min(lat/1000, 1)))**4


# ============================================================
def io_uring_available() -> bool:
    try:
        subprocess.check_output(
            ["fio", "--name=tmp", "--ioengine=io_uring", "--rw=read",
             "--size=1M", "--runtime=1", "--time_based", "--filename=/dev/null"],
            stderr=subprocess.STDOUT, timeout=5)
        return True
    except subprocess.CalledProcessError:
        return False
    except Exception:
        return False


def main():
    engines = ["psync", "libaio"]
    if io_uring_available():
        engines.append("io_uring")
    else:
        log.info("io_uring disable")

    param_space: list[dict[str, Any]] = [
        # CPU
        {"param": Integer("cpu_workers", 1, 1, psutil.cpu_count(logical=False) or 8), "expert_type": "CPU"},
        {"param": Integer("cpu_load",    50, 20, CPU_LOAD_MAX),                     "expert_type": "CPU"},
        # IO
        {"param": Integer("iodepth", 4, 1, 32),                                     "expert_type": "IO"},
        {"param": Integer("blocksize", 8, 4, 128),                                  "expert_type": "IO"},
        {"param": Integer("rwmix", 50, 0, 100),                                     "expert_type": "IO"},
    ]

    algo = MoEPPOAlgorithm(param_space)
    btl  = BottleneckIdentifier(cpu_thres=0.8, mem_thres=0.8,
                                io_thres=100.0, net_thres=50.0)
    algo.bottleneck_identifier = btl

    workload = MixedWorkload();  workload.btl = btl

    with open("moeppo_mixed.csv", "w", newline="") as fp:
        wr = csv.DictWriter(fp, fieldnames=[
            "iter", "tp_MB", "lat_ms", "reward", "b_cpu", "b_mem", "b_io", "b_net"])
        wr.writeheader()

        for it in range(ITERS):
            try:
                cfg = algo.predict()
                log.info(f"[{it}] cfg → {cfg}")

                bvec = workload.run(cfg)
                m    = workload.get_metrics()
                r    = reward(m)

                log.info(f"[{it}] tp={m['throughput']:.2f}MB/s  lat={m['latency']:.2f}ms  R={r:.4f}")
                algo.set_reward(r, m)

                wr.writerow(dict(iter=it, tp_MB=m["throughput"], lat_ms=m["latency"],
                                 reward=r, b_cpu=bvec[0], b_mem=bvec[1],
                                 b_io=bvec[2], b_net=bvec[3]))
                fp.flush()
            except KeyboardInterrupt:
                log.info("user interrupt")
                break
            except Exception:
                log.error("iteration error:\n"+traceback.format_exc())
                continue


if __name__ == "__main__":
    try:
        main()
    finally:
        subprocess.run(["pkill", "-9", "stress-ng", "fio"],
                       stderr=subprocess.DEVNULL, check=False)
