import pandas as pd

file_path = 'mongodb.xlsx'

throughput = pd.read_excel(file_path, usecols=['throughput'])
latency = pd.read_excel(file_path, usecols=['latency'])

throughput_sorted = throughput.sort_values(by='throughput', ascending=True)

latency_sorted = latency.sort_values(by = 'latency' , ascending=True)

p50_throughput = throughput_sorted.values[round(len(throughput_sorted)*0.5)]
p75_throughput = throughput_sorted.values[round(len(throughput_sorted)*0.75)]
p95_throughput = throughput_sorted.values[round(len(throughput_sorted)*0.95)]

p50_latency = latency_sorted.values[round(len(latency_sorted)*0.5)]
p75_latency = latency_sorted.values[round(len(latency_sorted)*0.75)]
p95_latency = latency_sorted.values[round(len(latency_sorted)*0.95)]

print(("p50_throughput ：{}，latency：{}").format(p50_throughput,p50_latency))
print(("p75_throughput ：{}，latency：{}").format(p75_throughput,p75_latency))
print(("p95_throughput ：{}，latency：{}").format(p95_throughput,p95_latency))

