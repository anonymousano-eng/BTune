import argparse
import logging
import subprocess
import time
import os
from modules.prediction.lstm_model import LSTM, DataProcessor
from modules.identification.bottleneck_identification import BottleneckIdentifier
from modules.tuning.rl_gate import GatedCollaborativeRL
from utils.sysctl_manager import ParameterManager
from utils.config_loader import ConfigLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(config_path):
    config_loader = ConfigLoader(config_path)
    config = {
        'expert_configs': config_loader.get_expert_configs(),
        'parameter_space': config_loader.get_parameter_space()
    }

    state_dim = config.get('state_dim', 64)
    lstm_model = LSTM(input_dim=state_dim, hidden_dim=128)
    data_processor = DataProcessor()
    metric_mapping = {
        'φcontext_switch': 0, 'ζmemcont': 1, 'σswap': 2, 'δdisk_latency': 3,
        'ηnet_queue': 4, 'σnet_bandwidth': 5, 'ωiowait': 6, 'ηdisk_queue': 7,
        'θcpu': 8, 'δnet': 9, 'ρretrans': 10, 'τresponse_time': 11, 'πthroughput': 12
    }
    expert_edges = [
        ('h_cpu', 'φcontext_switch'), ('h_cpu', 'θcpu'), ('h_mem', 'σswap'),
        ('h_io', 'δdisk_latency'), ('h_net', 'ηnet_queue'), ('h_net', 'σnet_bandwidth')
    ]
    bottleneck_identifier = BottleneckIdentifier(metric_mapping, expert_edges)

    rl_agent = GatedCollaborativeRL(config, state_dim)
    param_manager = ParameterManager(config)

    try:
        proc = subprocess.Popen(
            ["python3", "../A-Tune-Collector/collect_data.py", "-c", "../configs/collect_data.json"],
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )

        while True:
            line = proc.stdout.readline()
            if not line:
                time.sleep(0.1)
                continue

            system_state = data_processor.process_line(line)

            if system_state is not None:
                lstm_features = lstm_model.predict(system_state)
                bottleneck_vector = bottleneck_identifier.identify(lstm_features)
                actions = rl_agent.get_actions(system_state, lstm_features, bottleneck_vector)
                param_manager.apply_actions(actions)

    except KeyboardInterrupt:
        logger.info("Program terminated by user.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        if 'proc' in locals() and proc.poll() is None:
            proc.terminate()
        rl_agent.save_checkpoint('checkpoint.pth')
        logger.info("Model checkpoint saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BTune: System Parameter Tuning and Performance Monitoring')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()

    main(args.config)
    