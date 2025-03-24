from __future__ import annotations
from typing import Any, Mapping

import csv, io, logging, os, signal, subprocess, threading, time, traceback

import numpy as np
import psutil, redis

from oppertune.algorithms.moe_ppo_tuner import MoEPPOAlgorithm, BottleneckIdentifier
from oppertune.core.values import Integer, Categorical

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s| %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

HIGH_LOAD = dict(req=100_000 ,   
                 c=200,          
                 d=4096)          
BENCH_CMDS  = "get,set,hset,lpush,zadd,sadd"
ITERATIONS  = 1000
REDIS_CONF  = "redis.conf"

def safe_remove(*files: str):
    for f in files:
        try:
            if os.path.exists(f):
                os.remove(f)
                log.debug(f"rm {f}")
        except Exception as e:
            log.warning(f"remove {f} failed: {e}")

class Application:
    def __init__(self, conf_path: str = REDIS_CONF, err_log: str = "redis_error.log"):
        self.conf_path = conf_path
        self.err_log   = err_log
        self.redis_proc: subprocess.Popen | None = None
        self.metrics = {"throughput": 0.0, "latency": 1000.0}

    def set_parameters(self, ps: Mapping[str, Any]):
        with open(self.conf_path, "w") as fp:
            fp.write('save ""\nappendonly no\n')
            for k, v in ps.items():
                fp.write(f"{k} {v}\n")

    def run(self) -> list[float]:
        subprocess.run(["pkill", "-9", "redis-server"], stderr=subprocess.DEVNULL)
        safe_remove("dump.rdb", "appendonly.aof")
        time.sleep(0.5)

        with open(self.err_log, "a") as f:
            self.redis_proc = subprocess.Popen(
                ["redis-server", self.conf_path],
                stdout=subprocess.DEVNULL, stderr=f)
        time.sleep(2)
        if self.redis_proc.poll() is not None:
            log.error("Redis launch failed，see redis_error.log")
            return [0, 0, 0, 0]

        try:
            r = redis.Redis(host="localhost", port=6379, decode_responses=True)
            r.mset({f"k{i}": "v" for i in range(10)})
            r.close()
        except redis.RedisError:
            pass

        util_hist: list[float] = []
        def monitor():
            while self.redis_proc and self.redis_proc.poll() is None:
                util_hist.append(psutil.cpu_percent(0.2) / 100.0)
        threading.Thread(target=monitor, daemon=True).start()

        cmd = ["redis-benchmark",
               "-n", str(HIGH_LOAD["req"]),
               "-c", str(HIGH_LOAD["c"]),
               "-d", str(HIGH_LOAD["d"]),
               "--csv",
               "-t", BENCH_CMDS]
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            log.error(f"redis-benchmark failed:\n{e.output}")
            self._stop_redis()
            return [0, 0, 0, 0]

        self._stop_redis()
        safe_remove("dump.rdb", "appendonly.aof")

        tp, lat, lines = 0.0, 0.0, 0
        for row in csv.reader(io.StringIO(out)):
            if not row: continue
            try:
                tp += float(row[1])
                lat = max(lat, float(row[2]) if len(row) >= 3 else 0.5)
                lines += 1
            except (ValueError, IndexError):
                pass

        if lines:
            self.metrics = {"throughput": tp, "latency": lat}
        log.info(f"tp={tp:.0f} req/s  lat(max)={lat:.2f} ms  lines={lines}")

        cpu_avg = float(np.mean(util_hist)) if util_hist else 0.0
        return [1.0 if cpu_avg > 0.8 else 0.0, 0.0, 0.0, 0.0]

    def _stop_redis(self):
        if self.redis_proc and self.redis_proc.poll() is None:
            os.kill(self.redis_proc.pid, signal.SIGTERM)
            self.redis_proc.wait(5)
        subprocess.run(["pkill", "-9", "redis-server"],
                       stderr=subprocess.DEVNULL, check=False)

    def get_metrics(self): return self.metrics

def calc_reward(m: Mapping[str, Any]) -> float:
    return (m["throughput"] / 1_000_000) / (m["latency"] + 1)

def main():
    app = Application()
    btl = BottleneckIdentifier(cpu_thres=0.8, mem_thres=0.3,
                               io_thres=0.2, net_thres=0.2)
    app.bottleneck_identifier = btl

    params = [
        {'param': Integer("hz",            10, 10, 50),                      'expert_type': 'CPU'},
        {'param': Integer("tcp-keepalive", 300, 0, 3600),                    'expert_type': 'CPU'},
        {'param': Categorical("appendonly", "no", ["no", "yes"]),            'expert_type': 'IO'},
        {'param': Integer("tcp-backlog",   511, 128, 2048),                  'expert_type': 'NET'},
        {'param': Categorical("maxmemory-policy", "noeviction",
                              ["noeviction", "allkeys-lru", "volatile-lru"]), 'expert_type': 'NET'},
    ]
    algo = MoEPPOAlgorithm(params)
    algo.bottleneck_identifier = btl

    with open("moeppo_highload.csv", "w", newline="") as fp:
        wr = csv.DictWriter(fp, fieldnames=["it", "tp", "lat", "reward", "b_cpu"])
        wr.writeheader()

        for it in range(ITERATIONS):
            try:
                cfg = algo.predict()
                log.info(f"[{it:3d}] cfg → {cfg}")

                bvec = app.run()
                m    = app.get_metrics()
                r    = calc_reward(m)

                algo.set_reward(r, m)
                wr.writerow(dict(it=it, tp=m["throughput"], lat=m["latency"],
                                 reward=r, b_cpu=bvec[0]))
                fp.flush()

                log.info(f"[{it:3d}] reward={r:.4f}")
            except KeyboardInterrupt:
                log.info("Interrupted by user");  break
            except Exception:
                log.error(traceback.format_exc())
                continue

if __name__ == "__main__":
    try:
        main()
    finally:
        subprocess.run(["pkill", "-9", "redis-server"], stderr=subprocess.DEVNULL)
        safe_remove("dump.rdb", "appendonly.aof")
