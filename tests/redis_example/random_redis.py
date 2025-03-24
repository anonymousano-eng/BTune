#!/usr/bin/env python3
from __future__ import annotations
import csv, os, signal, subprocess, time, logging, redis, psutil, random
from typing import Any, Mapping, List, Dict

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TOTAL_REQS  = 100_000
CONCURRENCY = 200
DATA_SIZE   = 4096
BENCH_CMDS  = "get,set,hset,lpush,zadd,sadd"
ITERATIONS  = 1000
REDIS_CONF  = "redis.conf"

class SimpleRandomSearch:
    def __init__(self, param_space: List[Dict[str, Any]]):
        self.param_space = param_space
        self._req_id     = 0

    def _sample_param(self, spec: Dict[str, Any]):
        if spec["type"] == "int":
            return random.randint(spec["low"], spec["high"])
        if spec["type"] == "cat":
            return random.choice(spec["choices"])
        raise ValueError(f"Unknown param spec: {spec}")

    def predict(self):
        cfg = {spec["name"]: self._sample_param(spec) for spec in self.param_space}
        req_id = self._req_id
        self._req_id += 1
        return cfg, req_id

    def store_reward(self, req_id: int, reward: float):
        logger.debug(f"store_reward(id={req_id}, r={reward:.4f})")

    def set_reward(self, reward: float):
        logger.debug(f"set_reward(r={reward:.4f})")

def safe_remove(*files: str):
    for f in files:
        try:
            if os.path.exists(f):
                os.remove(f)
                logger.debug(f"rm {f}")
        except Exception as e:
            logger.warning(f"remove {f} failed: {e}")

class Application:
    def __init__(self, conf_path: str = REDIS_CONF):
        self.conf_path = conf_path
        self.metrics   = {"throughput": 0.0, "latency": 1000.0}

    def set_parameters(self, p: Mapping[str, Any]):
        with open(self.conf_path, "w") as fp:
            fp.write('save ""\nappendonly no\n')
            for k, v in p.items():
                fp.write(f"{k} {v}\n")

    def run(self):
        subprocess.run(["pkill", "-9", "redis-server"],
                       stderr=subprocess.DEVNULL, check=False)
        safe_remove("dump.rdb", "appendonly.aof")
        time.sleep(0.5)

        redis_proc = subprocess.Popen(["redis-server", self.conf_path],
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)
        time.sleep(2)
        if redis_proc.poll() is not None:
            logger.error("Redis failed to start")
            return

        try:
            r = redis.Redis(host="localhost", port=6379, decode_responses=True)
            r.mset({f"k{i}": "v" for i in range(10)})
            r.hset("h", "f", "v"); r.lpush("L", "v")
            r.zadd("Z", {"v": 1}); r.sadd("S", "v"); r.close()
        except redis.RedisError as e:
            logger.warning(f"Init redis data failed: {e}")

        cmd = ["redis-benchmark",
               "-n", str(TOTAL_REQS),
               "-c", str(CONCURRENCY),
               "-P", "1",
               "-d", str(DATA_SIZE),
               "--csv",
               "-t", BENCH_CMDS]
        try:
            bench_out = subprocess.check_output(cmd, text=True,
                                                stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            logger.error(f"redis-benchmark failed:\n{e.output}")
            self._stop_redis(redis_proc)
            return

        tp, lat, lines = 0.0, 0.0, 0
        for row in csv.reader(bench_out.splitlines()):
            if not row: 
                continue
            try:
                tp += float(row[1])
                lat = max(lat, float(row[2]) if len(row) >= 3 else 0.5)
                lines += 1
            except (ValueError, IndexError):
                pass
        self.metrics = {"throughput": tp, "latency": lat}
        logger.info(f"tp={tp:.0f} req/s  lat(max)={lat:.2f} ms  lines={lines}")

        cpu = psutil.cpu_percent(0.1)/100
        mem = psutil.virtual_memory().percent/100
        logger.info(f"CPU={cpu:.2f}  MEM={mem:.2f}")

        self._stop_redis(redis_proc)
        safe_remove("dump.rdb", "appendonly.aof")

    def _stop_redis(self, proc):
        if proc and proc.poll() is None:
            os.kill(proc.pid, signal.SIGTERM)
            proc.wait(5)
        subprocess.run(["pkill", "-9", "redis-server"],
                       stderr=subprocess.DEVNULL, check=False)

    def get_metrics(self):
        return self.metrics

def reward(m):
    return (m["throughput"]/1_000_000) / (m["latency"]+1)

def main():
    app = Application()

    params = [
        {"name": "hz",               "type": "int", "low": 10,  "high": 50},
        {"name": "tcp-backlog",      "type": "int", "low": 128, "high": 2048},
        {"name": "appendonly",       "type": "cat", "choices": ["no", "yes"]},
        {"name": "maxmemory-policy", "type": "cat", "choices": [
            "noeviction", "allkeys-lru", "allkeys-random", "volatile-lru"]},
        {"name": "tcp-keepalive",    "type": "int", "low": 0,   "high": 3600},
        {"name": "timeout",          "type": "int", "low": 0,   "high": 300},
    ]

    solver = SimpleRandomSearch(params)

    with open("random_search_result.csv", "w", newline="") as fp:
        wr = csv.DictWriter(fp, fieldnames=["it", "tp", "lat", "reward"])
        wr.writeheader()

        for it in range(ITERATIONS):
            cfg, req_id = solver.predict()
            logger.info(f"[{it}] config → {cfg}")
            app.set_parameters(cfg)
            app.run()
            m = app.get_metrics()
            r = reward(m)
            solver.store_reward(req_id, r)
            solver.set_reward(r)

            try:
                wr.writerow(dict(it=it, tp=m["throughput"], lat=m["latency"], reward=r))
                fp.flush()
            except OSError as e:
                logger.error(f"Write CSV failed: {e}")

            logger.info(f"[{it}] reward={r:.4f}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        subprocess.run(["pkill", "-9", "redis-server"], stderr=subprocess.DEVNULL)
        safe_remove("dump.rdb", "appendonly.aof")
