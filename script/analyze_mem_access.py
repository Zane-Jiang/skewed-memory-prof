#!/usr/bin/env python3

import re
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os

class Variable:
    def __init__(self, var_id, ptr, location, size, alloc_ts, free_ts):
        self.var_id = var_id
        self.start_addr = int(ptr, 16)
        self.end_addr = self.start_addr + int(size)
        self.size = int(size)
        self.alloc_ts = float(alloc_ts)
        self.free_ts = float(free_ts)
        self.location = location
        self.access_count = 0
        self.load_count = 0
        self.store_count = 0
        self.addr_stats = defaultdict(lambda: {'total': 0, 'load': 0, 'store': 0})

    def in_lifetime(self, ts):
        return self.alloc_ts <= ts <= self.free_ts

    def contains(self, addr):
        return self.start_addr <= addr < self.end_addr

    def record_access(self, addr, access_type):
        self.access_count += 1
        self.addr_stats[addr]['total'] += 1
        if access_type == 'load':
            self.load_count += 1
            self.addr_stats[addr]['load'] += 1
        elif access_type == 'store':
            self.store_count += 1
            self.addr_stats[addr]['store'] += 1

def load_variables(alloc_file):
    variables = []
    with open(alloc_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '' or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 6:
                continue  # Skip malformed lines
            var = Variable(
                var_id=parts[0],
                ptr=parts[1],
                location=parts[2],
                size=parts[3],
                alloc_ts=parts[4],
                free_ts=parts[5]
            )
            variables.append(var)
    return variables

def parse_perf_line(line):
    match = re.match(r'.*\s(\d+)\.(\d+):', line)
    if not match:
        return None
    ts_sec = int(match.group(1))
    ts_usec = int(match.group(2))
    timestamp = ts_sec * 1000 + ts_usec // 1000  # ms
    addr_match = re.search(r'([0-9a-f]{12,16})', line)
    if not addr_match:
        return None
    addr = int(addr_match.group(1), 16)
    if '|OP LOAD|' in line:
        access_type = 'load'
    elif '|OP STORE|' in line:
        access_type = 'store'
    else:
        return None
    return timestamp, addr, access_type

def plot_variable_access_heat(variables, output_prefix):
    import time
    import glob
    bar_dir = os.path.join(output_prefix, 'bar_plots')
    os.makedirs(bar_dir, exist_ok=True)
    # 清空旧图
    for f in glob.glob(os.path.join(bar_dir, '*.png')):
        os.remove(f)
    for idx, var in enumerate(variables):
        if var.access_count == 0:
            continue
        offsets = []
        counts = []
        for addr, stats in var.addr_stats.items():
            offset = addr - var.start_addr
            if 0 <= offset < var.size:
                offsets.append(offset)
                counts.append(stats['total'])
        offsets, counts = zip(*sorted(zip(offsets, counts))) if offsets else ([],[])
        plt.figure(figsize=(10, 4))
        plt.bar(offsets, counts, width=1.0)
        plt.xlabel('Offset in Variable (Byte)')
        plt.ylabel('Access Count')
        plt.title(f'Variable {var.var_id} @ {hex(var.start_addr)} Access Heat')
        plt.tight_layout()
        out_path = os.path.join(bar_dir, f'var_{var.var_id}_bar.png')
        plt.savefig(out_path)
        plt.close()

def plot_variable_access_colormap(variables, output_prefix):
    import time
    import glob
    heatmap_dir = os.path.join(output_prefix, 'heatmaps')
    os.makedirs(heatmap_dir, exist_ok=True)
    for f in glob.glob(os.path.join(heatmap_dir, '*.png')):
        os.remove(f)
    for idx, var in enumerate(variables):
        if var.access_count == 0:
            continue
        offsets = []
        counts = []
        for addr, stats in var.addr_stats.items():
            offset = addr - var.start_addr
            if 0 <= offset < var.size:
                offsets.append(offset)
                counts.append(stats['total'])
        plt.figure(figsize=(8, 4))
        plt.scatter([0]*len(offsets), offsets, c=counts, cmap='coolwarm', marker='s', s=8, linewidths=0)
        plt.colorbar(label='Access Count')
        plt.title(f'Var {var.var_id}\n{hex(var.start_addr)} ~ {hex(var.end_addr-1)}\nLife: {var.alloc_ts}~{var.free_ts}')
        plt.xlabel('Variable')
        plt.ylabel('Offset (Byte)')
        plt.tight_layout()
        out_path = os.path.join(heatmap_dir, f'var_{var.var_id}_heatmap.png')
        plt.savefig(out_path)
        plt.close()

def analyze(perf_file, alloc_file, output_file):
    variables = load_variables(alloc_file)

    with open(perf_file, 'r') as f:
        for line in f:
            parsed = parse_perf_line(line)
            if parsed:
                ts, addr, access_type = parsed
                for var in variables:
                    if var.in_lifetime(ts) and var.contains(addr):
                        var.record_access(addr, access_type)

    with open(output_file, 'w') as out:
        out.write('===== Summary: Variable Memory Access Statistics =====\n')
        out.write('Start Address\tSize(Bytes)\tLifetime(ms)\tTotal Accesses\tLoad Count\tStore Count\n')
        for var in variables:
            out.write(f'{hex(var.start_addr)}\t{var.size}\t{var.alloc_ts}~{var.free_ts}\t{var.access_count}\t{var.load_count}\t{var.store_count}\n')

        out.write('\n===== Detailed Accesses per Variable =====\n')
        for var in variables:
            if var.access_count == 0:
                continue
            out.write(f'\nVariable Start Address: {hex(var.start_addr)}\n')
            out.write('Accessed Address\tTotal\tLoads\tStores\n')
            for addr, stats in sorted(var.addr_stats.items()):
                out.write(f'{hex(addr)}\t{stats["total"]}\t{stats["load"]}\t{stats["store"]}\n')

    plot_variable_access_heat(variables, "result")
    plot_variable_access_colormap(variables, "result")


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python3 analyze_mem_access.py <perf_output.txt> <alloc_info.txt> <output.txt>")
        sys.exit(1)

    perf_file = sys.argv[1]
    alloc_file = sys.argv[2]
    output_file = sys.argv[3]

    analyze(perf_file, alloc_file, output_file)
