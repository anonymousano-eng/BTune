import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
import logging
import itertools
from scipy.stats import chi2_contingency

logging.getLogger('pgmpy').setLevel(logging.ERROR)

class BottleneckIdentifier:
    def __init__(self, metric_mapping, expert_edges=None):
        self.node_hierarchy = {
            'root': ['h_cpu', 'h_mem', 'h_io', 'h_net'],
            'mid': ['φcontext_switch', 'ζmemcont', 'σswap', 'δdisk_latency', 'ηnet_queue', 'σnet_bandwidth', 'ωiowait', 'ηdisk_queue'],
            'downstream': ['θcpu', 'δnet', 'ρretrans'],
            'global': ['τresponse_time', 'πthroughput']
        }
        self.metric_map = metric_mapping
        self.model = BayesianNetwork()
        self._build_network(expert_edges)
        self._init_all_cpds()
        self._validate_model()
        self.infer_engine = VariableElimination(self.model)
        self.alpha = 0.01
        self.dirichlet_prior = 0.1
        self.state_names = {0: 'low', 1: 'normal', 2: 'high'}

    def _build_network(self, expert_edges):
        all_nodes = []
        for tier in self.node_hierarchy.values():
            all_nodes.extend(tier)
        self.model.add_nodes_from(all_nodes)
        core_edges = [
            ('h_cpu', 'φcontext_switch'), ('h_cpu', 'θcpu'),
            ('h_mem', 'ζmemcont'), ('h_io', 'δdisk_latency'),
            ('h_net', 'ηnet_queue'), ('h_net', 'σnet_bandwidth'),
            ('ζmemcont', 'σswap'), ('σswap', 'ωiowait'),
            ('δdisk_latency', 'ηdisk_queue'), ('ηnet_queue', 'δnet'),
            ('δnet', 'ρretrans'), ('ωiowait', 'ηdisk_queue'),
            ('ηdisk_queue', 'τresponse_time'), ('θcpu', 'τresponse_time'),
            ('σnet_bandwidth', 'τresponse_time'), ('ρretrans', 'τresponse_time'),
            ('τresponse_time', 'πthroughput'), ('σnet_bandwidth', 'πthroughput')
        ]
        self.model.add_edges_from(core_edges)
        if expert_edges:
            self.model.add_edges_from(expert_edges)

    def _init_all_cpds(self):
        for node in nx.topological_sort(self.model):
            self._init_single_cpd(node)

    def _init_single_cpd(self, node):
        if self.model.get_cpds(node):
            self.model.remove_cpds(node)
        parents = list(self.model.predecessors(node))
        card = 3 if node in self.node_hierarchy['mid'] else 2
        states = ['low', 'normal', 'high'][:card]
        state_names = {node: states}
        if parents:
            parent_cards = []
            for p in parents:
                if not self.model.get_cpds(p):
                    self._init_single_cpd(p)
                parent_cards.append(self.model.get_cpds(p).variable_card)
                state_names[p] = ['low', 'normal', 'high'][:self.model.get_cpds(p).variable_card]
            values_shape = (card,) + tuple(parent_cards)
            values = np.ones(values_shape) / card
            values_2d = values.reshape(card, -1).tolist()
            cpd = TabularCPD(variable=node, variable_card=card, values=values_2d, evidence=parents, evidence_card=parent_cards, state_names=state_names)
        else:
            cpd = TabularCPD(variable=node, variable_card=card, values=[[1/card] for _ in range(card)], state_names=state_names)
        self.model.add_cpds(cpd)

    def _validate_model(self):
        self.model.check_model()

    def dynamic_structure_learning(self, data):
        from pgmpy.estimators import PC
        pc = PC(data)
        estimated_edges = pc.estimate(return_type="dag").edges()
        for edge in estimated_edges:
            if edge not in self.model.edges():
                temp_model = self.model.copy()
                temp_model.add_edge(*edge)
                try:
                    nx.find_cycle(temp_model)
                    logging.warning(f"Adding edge {edge} would create a cycle. Skipping.")
                    continue
                except nx.NetworkXNoCycle:
                    contingency = pd.crosstab(data[edge[0]], data[edge[1]])
                    _, p, _, _ = chi2_contingency(contingency)
                    if p < self.alpha * 0.5:
                        self.model.add_edge(*edge)
        self._init_all_cpds()

    def manual_update_cpds(self, training_data):
        for node in self.model.nodes():
            parents = list(self.model.predecessors(node))
            card = self.model.get_cpds(node).variable_card
            if not parents:
                counts = np.zeros(card)
                for i in range(card):
                    counts[i] = (training_data[node] == i).sum()
                counts += self.dirichlet_prior
                counts = counts / counts.sum()
                values = counts.reshape(card, 1).tolist()
                cpd = TabularCPD(variable=node, variable_card=card, values=values)
            else:
                parent_cards = [self.model.get_cpds(p).variable_card for p in parents]
                parent_state_combinations = list(itertools.product(*[range(pc) for pc in parent_cards]))
                values = np.zeros((card, len(parent_state_combinations)))
                for j, comb in enumerate(parent_state_combinations):
                    cond = np.ones(len(training_data), dtype=bool)
                    for idx, p in enumerate(parents):
                        cond &= (training_data[p] == comb[idx])
                    for i in range(card):
                        values[i, j] = (training_data[node][cond] == i).sum()
                    values[:, j] += self.dirichlet_prior
                    col_sum = values[:, j].sum()
                    if col_sum > 0:
                        values[:, j] /= col_sum
                values_list = values.tolist()
                cpd = TabularCPD(variable=node, variable_card=card, values=values_list, evidence=parents, evidence_card=parent_cards)
            self.model.add_cpds(cpd)

    def advanced_discretization(self, lstm_output):
        discretized = {}
        for metric, idx in self.metric_map.items():
            mu = lstm_output['means'][idx]
            sigma = np.sqrt(lstm_output['vars'][idx])
            card = self.model.get_cpds(metric).variable_card
            if card == 3:
                discretized[metric] = 'low' if mu < 0.3 else 'normal' if mu < 0.7 else 'high'
            else:
                discretized[metric] = 'low' if mu < 0 else 'normal'
        print("Discretized Evidence:", discretized)
        return discretized

    def causal_impact_analysis(self, evidence):
        str_to_int = {v: k for k, v in self.state_names.items()}
        formatted_evidence = {k: str_to_int[v] for k, v in evidence.items()}
        try:
            posterior = self.infer_engine.query(variables=self.node_hierarchy['root'], evidence=formatted_evidence, joint=False)
        except Exception as e:
            logging.error(f"Error in causal_impact_analysis: {e}")
            logging.error(f"Evidence causing the error: {formatted_evidence}")
            raise
        impact_scores = {}
        for bottleneck in self.node_hierarchy['root']:
            descendants = list(nx.descendants(self.model, bottleneck))
            sensitivity = sum(np.abs(self.model.get_cpds(desc).values * 1.1 - self.model.get_cpds(desc).values).mean() for desc in descendants if self.model.get_cpds(desc) is not None)
            impact_scores[bottleneck] = {
                'posterior': posterior[bottleneck].values,
                'sensitivity': sensitivity,
                'score': np.mean(posterior[bottleneck].values) * sensitivity
            }
        return self._trace_causal_chains(impact_scores)

    def _trace_causal_chains(self, scores):
        chains = []
        for bottleneck in self.node_hierarchy['root']:
            chain = [bottleneck]
            current = bottleneck
            visited = set()
            while True:
                children = [c for c in self.model.successors(current) if c not in visited]
                if not children:
                    break
                best_child = max(children, key=lambda c: scores.get(c, {}).get('score', 0))
                chain.append(best_child)
                visited.add(current)
                current = best_child
                if len(chain) >= 5:
                    break
            chains.append((bottleneck, chain))
        return sorted(chains, key=lambda x: scores[x[0]]['score'], reverse=True)

if __name__ == "__main__":
    metric_mapping = {
        'φcontext_switch': 0, 'ζmemcont': 1, 'σswap': 2, 'δdisk_latency': 3,
        'ηnet_queue': 4, 'σnet_bandwidth': 5, 'ωiowait': 6, 'ηdisk_queue': 7,
        'θcpu': 8, 'δnet': 9, 'ρretrans': 10, 'τresponse_time': 11, 'πthroughput': 12
    }
    expert_edges = [
        ('h_cpu', 'φcontext_switch'), ('h_cpu', 'θcpu'), ('h_mem', 'σswap'),
        ('h_io', 'δdisk_latency'), ('h_net', 'ηnet_queue'), ('h_net', 'σnet_bandwidth')
    ]
    all_nodes = list(set(['h_cpu', 'h_mem', 'h_io', 'h_net'] + list(metric_mapping.keys())))
    n_samples = 1000
    synthetic_dict = {}
    mid_nodes = set(['φcontext_switch', 'ζmemcont', 'σswap', 'δdisk_latency', 'ηnet_queue', 'σnet_bandwidth', 'ωiowait', 'ηdisk_queue'])
    for node in all_nodes:
        if node in mid_nodes:
            synthetic_dict[node] = np.random.choice([0, 1, 2], size=n_samples)
        else:
            synthetic_dict[node] = np.random.choice([0, 1], size=n_samples)
    synthetic_data = pd.DataFrame(synthetic_dict)
    try:
        bn = BottleneckIdentifier(metric_mapping, expert_edges)
        bn.dynamic_structure_learning(synthetic_data)
        bn.manual_update_cpds(synthetic_data)
        lstm_output = {'means': np.random.randn(13), 'vars': np.abs(np.random.randn(13))}
        evidence = bn.advanced_discretization(lstm_output)
        results = bn.causal_impact_analysis(evidence)
        for bottleneck, chain in results:
            print(f"{bottleneck}: {'→'.join(chain)}")
    except Exception as e:
        print(f"Initialization failed: {str(e)}")
