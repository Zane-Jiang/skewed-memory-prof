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
    
    print(f"[INFO] Starting to generate bar plots in '{bar_dir}'...")
    total_start_time = time.time()

    for f in glob.glob(os.path.join(bar_dir, '*.png')):
        os.remove(f)


    for idx, var in enumerate(variables):
        if var.access_count == 0:
            continue
            
        var_start_time = time.time()
        
        data_prep_start_time = time.time()
        
        # Define thresholds
        page_size = 4 * 1024  # 4KB
        max_bins = 1000
        very_large_threshold = page_size * max_bins # 4MB

        plt.xlabel('Offset in Variable (Byte)')
        if var.size <= page_size:
            continue
        print(f"\n[{idx+1}/{len(variables)}] Processing variable id={var.var_id}, size={var.size} bytes...")

        if var.size <= very_large_threshold:
            print(f"    -> Mode: Binning to {page_size//1024}KB pages (size <= {very_large_threshold/1024/1024}MB).")
            
            bin_size = page_size
            num_bins = (var.size + bin_size - 1) // bin_size
            bin_counts = [0] * num_bins
            for addr, stats in var.addr_stats.items():
                offset = addr - var.start_addr
                if 0 <= offset < var.size:
                    bin_idx = offset // bin_size
                    bin_counts[bin_idx] += int(stats['total'])
            
            print(f"    -> Data preparation took: {time.time() - data_prep_start_time:.3f}s")
            plotting_start_time = time.time()

            bin_offsets = [i * bin_size for i in range(num_bins)]
            plt.figure(figsize=(12, 4))
            plt.bar(bin_offsets, bin_counts, width=bin_size, align='edge')
            
            plt.title(f'Variable {var.var_id} @ {hex(var.start_addr)} Access Heat ({bin_size//1024}KB Bins)')
            plt.ylabel('Access Count per 4KB Page')
            
            x_ticks = [0]
            if num_bins > 1:
                step = max(1, num_bins // 10)
                x_ticks.extend([i * bin_size for i in range(step, num_bins, step)])
            x_ticks.append(var.size)
            plt.xticks(sorted(list(set(x_ticks))))
            x_labels = [f"{int(x/1024)}K" for x in plt.gca().get_xticks()]
            if len(x_labels) > 0:
                x_labels[-1] = f"{int(var.size/1024)}K\n({hex(var.end_addr)})"
            plt.gca().set_xticklabels(x_labels)

            if any(bin_counts):
                ymax = max(bin_counts) * 1.1
                plt.ylim(0, ymax if ymax > 0 else 1)
                plt.yticks(range(0, int(ymax) + 1, max(1, int(ymax) // 8)))
                
        else:
            print(f"    -> Mode: Binning to {max_bins} bins (size > {very_large_threshold/1024/1024}MB).")

            num_bins = max_bins
            bin_size = (var.size + num_bins - 1) // num_bins
            bin_counts = [0] * num_bins
            for addr, stats in var.addr_stats.items():
                offset = addr - var.start_addr
                if 0 <= offset < var.size:
                    bin_idx = min(offset // bin_size, num_bins - 1)
                    bin_counts[bin_idx] += int(stats['total'])

            print(f"    -> Data preparation took: {time.time() - data_prep_start_time:.3f}s")
            plotting_start_time = time.time()

            bin_offsets = [i * bin_size for i in range(num_bins)]
            plt.figure(figsize=(12, 4))
            plt.bar(bin_offsets, bin_counts, width=bin_size, align='edge')

            plt.title(f'Variable {var.var_id} @ {hex(var.start_addr)} Access Heat ({num_bins} Bins)')
            plt.ylabel(f'Access Count per Bin (~{bin_size/1024:.1f}KB)')

            x_ticks = [0]
            if num_bins > 1:
                step = max(1, num_bins // 10)
                x_ticks.extend([i * bin_size for i in range(step, num_bins, step)])
            x_ticks.append(var.size)
            plt.xticks(sorted(list(set(x_ticks))))
            x_labels = [f"{int(x/1024)}K" for x in plt.gca().get_xticks()]
            if len(x_labels) > 0:
                x_labels[-1] = f"{int(var.size/1024)}K\n({hex(var.end_addr)})"
            plt.gca().set_xticklabels(x_labels)

            if any(bin_counts):
                ymax = max(bin_counts) * 1.1
                plt.ylim(0, ymax if ymax > 0 else 1)
                plt.yticks(range(0, int(ymax) + 1, max(1, int(ymax) // 8)))
                
        # Common part for all plots
        plt.xlim(0, var.size)
        
        total_accesses = var.access_count
        size_mb = var.size / (1024 * 1024)
        plt.figtext(0.5, 0.01, f"Total Accesses: {total_accesses}, Size: {size_mb:.3f} MB, Location: {var.location}", ha="center", fontsize=12, bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5})
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        out_path = os.path.join(bar_dir, f'var_{var.var_id}_bar.png')
        plt.savefig(out_path)
        plt.close()

        print(f"    -> Plotting & saving took: {time.time() - plotting_start_time:.3f}s")
        print(f"    -> Finished in {time.time() - var_start_time:.3f}s. Saved to: {out_path}")

    print(f"\n[INFO] Finished generating all plots in {time.time() - total_start_time:.3f}s.")


sys.path.append('/home/jz/.local/lib/python3.13/site-packages')
from intervaltree import IntervalTree

def build_interval_tree(variables):
    tree = IntervalTree()
    for var in variables:
        if var.size > 0: 
            tree[var.start_addr:var.end_addr] = var
    return tree

def analyze(perf_file, alloc_file, output_file):
    import time
    total_start_time = time.time()
    variables = load_variables(alloc_file)
    interval_tree = build_interval_tree(variables)

    with open(perf_file, 'r') as f:
        for line in f:
            parsed = parse_perf_line(line)
            if parsed:
                ts, addr, access_type = parsed
                for iv in interval_tree[addr]:
                    var = iv.data
                    if var.in_lifetime(ts):
                        var.record_access(addr, access_type)
    print(f"[INFO] Finished analyzing memory access in {time.time() - total_start_time:.3f}s.")
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
    plot_all_variables_access_cdf(variables, "result")

def plot_all_variables_access_cdf(variables, output_prefix):
    import time
    import os
    import glob
    os.makedirs(output_prefix, exist_ok=True)
    out_path = os.path.join(output_prefix, 'all_vars_cdf.png')

    print(f"[INFO] Generating combined CDF plot for all variables...")

    plt.figure(figsize=(10, 6))
    page_size = 4 * 1024
    debug_dir = os.path.join(output_prefix, "page_debug")
    os.makedirs(debug_dir, exist_ok=True)
    for f in glob.glob(os.path.join(debug_dir, '*.txt')):
        os.remove(f)

    for idx, var in enumerate(variables):
        if var.access_count == 0:
            continue
        if var.size < page_size:
            continue
        num_pages = (var.size + page_size - 1) // page_size
        page_accesses = [0] * num_pages

        for addr, stats in var.addr_stats.items():
            offset = addr - var.start_addr
            if 0 <= offset < var.size:
                page_idx = offset // page_size
                page_accesses[page_idx] += stats['total']

        sorted_accesses = sorted(page_accesses, reverse=True)
        total_access = sum(sorted_accesses)
        if total_access == 0:
            continue

        num_pages = len(sorted_accesses)
        page_percent = np.concatenate([[0], np.arange(1, num_pages + 1) / num_pages * 100])
        cumsum_access_percent = np.concatenate([[0], np.cumsum(sorted_accesses) / total_access * 100])
     
        #label 显示到图的下方，图的大小不固定
        plt.figtext(0.5, 0.01, f"Var {var.var_id}{var.location}({var.size//1024}KB)", ha="center", fontsize=12, bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5})
        # label = f'Var {var.var_id}{var.location}({var.size//1024}KB)'
        plt.plot(page_percent, cumsum_access_percent,  linewidth=1.5)
        
        debug_file = os.path.join(debug_dir, f'var_{var.var_id}_pages.txt')

        with open(debug_file, 'w') as df:
            df.write(f'# Variable {var.var_id}, Size={var.size} bytes, Pages={num_pages}\n')
            df.write('Page_Index_Sorted\tAccess_Count\tAccess_Percent\n')
            for rank, count in enumerate(sorted_accesses):
                percent = count / total_access * 100
                df.write(f'{rank}\t{count}\t{percent:.2f}\n')

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xticks(np.arange(0, 110, 10))
    plt.yticks(np.arange(0, 110, 10))
    plt.xlabel('Page Percentage (Sorted by Access Count Desc)', fontsize=12)
    plt.ylabel('Cumulative Access Percentage', fontsize=12)
    plt.title('Access Distribution CDF per Variable', fontsize=14)
    plt.grid(True)
    plt.legend(loc='lower right', fontsize=10, framealpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()





    print(f"[INFO] Saved combined CDF plot to {out_path}")


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python3 analyze_mem_access.py <perf_output.txt> <alloc_info.txt> <output.txt>")
        sys.exit(1)
    perf_file = sys.argv[1]
    alloc_file = sys.argv[2]
    output_file = sys.argv[3]
    analyze(perf_file, alloc_file, output_file)
