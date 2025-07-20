#!/usr/bin/env python3

import re
import sys
from collections import defaultdict

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
    match = re.match(r'.*\s(\d+)\.(\d+):\s+\d+\s+cpu/mem-(loads|stores)', line)
    if match:
        ts_sec = int(match.group(1))
        ts_usec = int(match.group(2))
        access_type = match.group(3)
        timestamp = ts_sec * 1000 + ts_usec // 1000  # convert to ms
        addr_match = re.search(r'([0-9a-f]{12,16})', line)
        if addr_match:
            addr = int(addr_match.group(1), 16)
            return timestamp, addr, access_type
    return None

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


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python3 analyze_mem_access.py <perf_output.txt> <alloc_info.txt> <output.txt>")
        sys.exit(1)

    perf_file = sys.argv[1]
    alloc_file = sys.argv[2]
    output_file = sys.argv[3]

    analyze(perf_file, alloc_file, output_file)
