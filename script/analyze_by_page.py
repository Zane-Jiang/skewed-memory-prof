#!/usr/bin/env python3

import re
import sys
from collections import defaultdict
import os
import time
import glob
from collections import OrderedDict
import csv
import numpy as np
import bisect
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

events = ["cycles", "CYCLE_ACTIVITY.STALLS_L3_MISS", 
  "OFFCORE_REQUESTS.DEMAND_DATA_RD", 
  "OFFCORE_REQUESTS_OUTSTANDING.CYCLES_WITH_DEMAND_DATA_RD"]
demand_data_rd_idx = events.index("OFFCORE_REQUESTS.DEMAND_DATA_RD")
cyc_demand_data_rd_idx = events.index("OFFCORE_REQUESTS_OUTSTANDING.CYCLES_WITH_DEMAND_DATA_RD")
l3_stall_idx = events.index("CYCLE_ACTIVITY.STALLS_L3_MISS")
cyc_idx = events.index("cycles")

PAGE_SIZE = 2*1024*1024
class Page:
    def __init__(self, idx, start_addr):
        self.access_count = 0
        self.idx = idx
        self.start_addr = start_addr
        self.access_ratio = 0.0
        self.score = 0.0
        self.time_slice_access_counts = []
        self.time_slice_access_ratios = []
        
    def contains(self, addr):
        return self.start_addr <= addr <= (self.start_addr+PAGE_SIZE)
        
class Variable:
    def __init__(self, var_id, ptr, location, size, alloc_ts, free_ts,index):
        self.var_id = var_id
        self.start_addr = int(ptr, 16)
        self.end_addr = self.start_addr + int(size)
        self.size = int(size)
        self.alloc_ts = int(alloc_ts)
        self.free_ts = int(free_ts)
        self.location = location
        self.index = index
        self.score = 0.0

        end_pg   = (self.size) // PAGE_SIZE
        self.pages = []
        for p in range(0, end_pg + 1):
            self.pages.append(Page(idx=p, start_addr=self.start_addr + p * PAGE_SIZE))

    def calculate_score(self):
        self.score = sum(page.score for page in self.pages)
        self.score = self.score / self.size if self.size > 0 else 0.0

    def in_lifetime(self, ts):
        return self.alloc_ts <= ts <= self.free_ts

    def contains(self, addr):
        return self.start_addr <= addr < self.end_addr

class File_Paser:
    @staticmethod
    def load_variables(alloc_file):
        variables = []
        with open(alloc_file, 'r') as f:
            
            index = 0
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
                    free_ts=parts[5],
                    index=index
                )
                index += 1
                variables.append(var)
        
        # 计算开始和结束时间
        if variables:
            start_time = min(var.alloc_ts for var in variables)
            end_time = max(var.free_ts for var in variables)
        else:
            start_time = 0
            end_time = 0
            
        return start_time,end_time,variables
    

    def __parse_peb_line(line):
        # Match L3 miss stall events
        if 'pebs:pebs' not in line:
            return None
        arr = re.split(' |,|\n', line)
        arr = [x for x in arr if len(x) > 0]
        address = int(arr[8], 16)
        time = int(arr[10], 16)
        return time, address

    @staticmethod
    def load_pebs_file(variables,peb_file):
        flow_time_addr = []
        with open(peb_file, 'r') as f:
            for line in f:
                parsed = File_Paser.__parse_peb_line(line)
                if parsed:
                    flow_time_addr.append(parsed)
        return flow_time_addr

    @staticmethod 
    def load_stall_perf_file(file_name):
        perf_data = OrderedDict()
        for event in events:
            perf_data[event] = []

        with open(file_name) as csv_file:
            csv_header = csv.reader(csv_file, delimiter=' ')
            line_count = 0
            for row in csv_header:
                line_count += 1
                processed_row = [item for item in row if item]
                for event in events:
                    if event in processed_row:
                        value = float(processed_row[1].replace(',', ''))
                        perf_data[event].append(value)
                
        demand_data_rd = np.asarray(perf_data[events[demand_data_rd_idx]])
        cyc_demand_data_rd = np.asarray(perf_data[events[cyc_demand_data_rd_idx]])
        l3_stall = np.asarray(perf_data[events[l3_stall_idx]])
        cyc = np.asarray(perf_data[events[cyc_idx]])        

        a = 24.67
        b = 0.87
        l3_stall_per_cyc = l3_stall/cyc 
        aol = cyc_demand_data_rd/demand_data_rd  #aol
        slowdown = [l3_stall_per_cyc[i]/(a/aol[i]+b) for i in range(len(aol))] ## 预测减速

        flow_performance_pridiction = aol,slowdown
        return flow_performance_pridiction


def _process_chunk(args):
    chunk, thresholds, simple_vars = args
    num_t = len(thresholds)
    from collections import defaultdict
    local_counts = defaultdict(lambda: np.zeros(num_t, dtype=np.int64))

    for time_val, address in chunk:
        idx = bisect.bisect_right(thresholds, time_val)
        if idx >= num_t:
            idx = num_t - 1

        for v_idx, (alloc_ts, free_ts, var_start, var_end, page_starts) in enumerate(simple_vars):
            if not (alloc_ts <= time_val <= free_ts):
                continue
            if not (var_start <= address < var_end):
                continue

            page_idx = (address - var_start) // PAGE_SIZE
            if 0 <= page_idx < len(page_starts):
                local_counts[(v_idx, int(page_idx))][idx] += 1

    return {k: v.tolist() for k, v in local_counts.items()}


def get_access_ratio_flow(new_ts, flow_variables, flow_time_addr):
    for variable in flow_variables:
        for page in variable.pages:
            page.time_slice_access_counts = [0] * len(new_ts)
            page.time_slice_access_ratios = [0.0] * len(new_ts)

    if not flow_time_addr:
        return

    # 构建轻量级变量元数据，便于多进程传输
    simple_vars = []
    for variable in flow_variables:
        page_starts = [p.start_addr for p in variable.pages]
        simple_vars.append((
            variable.alloc_ts,
            variable.free_ts,
            variable.start_addr,
            variable.end_addr,
            page_starts,
        ))

    num_workers = max(1, (multiprocessing.cpu_count() or 1))
    chunk_size = max(1, len(flow_time_addr) // (num_workers * 4) or 1)
    chunks = [flow_time_addr[i:i + chunk_size] for i in range(0, len(flow_time_addr), chunk_size)]

    aggregated = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_chunk, (chunk, new_ts, simple_vars)) for chunk in chunks]
        for fut in as_completed(futures):
            part = fut.result()
            for key, arr in part.items():
                if key not in aggregated:
                    aggregated[key] = np.array(arr, dtype=np.int64)
                else:
                    aggregated[key] += np.array(arr, dtype=np.int64)

    for (v_idx, p_idx), counts in aggregated.items():
        page = flow_variables[v_idx].pages[p_idx]
        page.time_slice_access_counts = counts.tolist()

    time_slice_total_access_counts = np.zeros(len(new_ts), dtype=np.int64)
    for variable in flow_variables:
        for page in variable.pages:
            time_slice_total_access_counts += np.array(page.time_slice_access_counts, dtype=np.int64)

    total_sum = int(time_slice_total_access_counts.sum())

    # 计算每页各时间片的访问比例与整体占比
    for variable in flow_variables:
        for page in variable.pages:
            if time_slice_total_access_counts.any():
                ratios = np.divide(
                    np.array(page.time_slice_access_counts, dtype=np.float64),
                    time_slice_total_access_counts,
                    out=np.zeros_like(time_slice_total_access_counts, dtype=np.float64),
                    where=time_slice_total_access_counts > 0,
                )
                page.time_slice_access_ratios = ratios.tolist()
            else:
                page.time_slice_access_ratios = [0.0] * len(new_ts)

            page.access_count = int(sum(page.time_slice_access_counts))
            page.access_ratio =  (page.access_count / total_sum) if total_sum > 0 else 0.0

    return


def _score_page(args):
    v_idx, p_idx, alloc_ts, free_ts, ratios, new_ts, aol, slowdown = args
    page_time_scores = []
    if v_idx == 4:
        print(f"[DEBUG] Scoring Variable {v_idx} Page {p_idx}, alloc_ts: {alloc_ts}, free_ts: {free_ts}")
    for i in range(len(new_ts)):
        ts_val = new_ts[i]
        if not (alloc_ts <= ts_val <= free_ts):
            continue
        factor = 1
        min_ratio = 0.000003
        max_ratio = 0.0007
        ai = aol[i] if i < len(aol) else aol[-1]
        if ai < 80 and ai > 60:
            factor = 2
            min_ratio = 0.4
            max_ratio = 0.6
        elif ai <= 60 and ai > 45:
            factor = 4
        elif ai > 40:
            factor = 8
        else:
            factor = 12
        access_ratio = ratios[i] if i < len(ratios) else 0.0
        if access_ratio >= max_ratio:
            time_slice_score = access_ratio * slowdown[i] / factor
        elif access_ratio >= min_ratio:
            time_slice_score = access_ratio * slowdown[i]
        else:
            time_slice_score = access_ratio * slowdown[i] * factor
        # page_time_scores.append(time_slice_score)
        page_time_scores.append(access_ratio*1)
    score = sum(page_time_scores) if page_time_scores else 0.0
    if v_idx == 4:
        print(f"[DEBUG] Variable {v_idx} Page {p_idx} Score: {score}")
    return (v_idx, p_idx, score)


def get_page_sore_flow(new_ts,flow_variables, flow_performance_pridiction):
    aol, slowdown = flow_performance_pridiction
    tasks = []
    for v_idx, variable in enumerate(flow_variables):
        for p_idx, page in enumerate(variable.pages):
            tasks.append((
                v_idx,
                p_idx,
                variable.alloc_ts,
                variable.free_ts,
                page.time_slice_access_ratios,
                new_ts,
                aol,
                slowdown,
            ))
    if not tasks:
        return flow_variables
    num_workers = max(1, (multiprocessing.cpu_count() or 1))
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 均匀切块分配，避免任务过小
        chunk = max(1, len(tasks)//(num_workers*4) or 1)
        for res in executor.map(_score_page, tasks, chunksize=chunk):
            results.append(res)
    for v_idx, p_idx, score in results:
        flow_variables[v_idx].pages[p_idx].score = score
    return flow_variables

def _plot_variable_batch(args):
    variable_batch, output_dir = args
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    saved_paths = []
    for var_id, page_scores in variable_batch:
        if not page_scores:
            continue
            
        page_indices = list(range(len(page_scores)))
        figure_width = min(max(len(page_scores) * 0.25 + 2.0, 6.0), 32.0)
        
        fig = plt.figure(figsize=(figure_width, 4.5), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(page_indices, page_scores, color='#4C78A8', width=0.8)
        ax.set_xlabel('Subpage Index')
        ax.set_ylabel('Page Score (x100)')
        ax.set_title(f'Variable index{var_id} ')
        
        safe_var_id = str(var_id).replace('/', '_').replace('\\', '_').replace(' ', '_')
        out_path = os.path.join(output_dir, f"var_{safe_var_id}_page_scores.png")
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        print(f"[INFO] Saved plot for variable {var_id} to {out_path}") 
        saved_paths.append(out_path)
    
    return saved_paths

# def merge_value(flow_variables):
#     ret = set()
#     id_map = {}
#     for variable in flow_variables:
#         if variable.var_id not in id_map:
#             variable.calculate_score()
#             id_map[variable.var_id] = variable
#     #sora 这里存在bug，相同的id就使用了相同的variable对象，导致前边score被覆盖，用一个socre表示所有socre
#     ret = set(id_map.values())
#     return ret


def plot_variable_page_scores(flow_variables, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    plot_data = []
    for variable in flow_variables:
        if variable.size <= PAGE_SIZE:
            continue
        scores = [page.score * 100.0 for page in variable.pages]
        if scores:
            plot_data.append((variable.index, scores))
    
    if not plot_data:
        return
    
    num_workers = min(8, (multiprocessing.cpu_count() or 1))
    batch_size = max(1, len(plot_data) // num_workers)
    batches = [plot_data[i:i + batch_size] for i in range(0, len(plot_data), batch_size)]
    
    num_workers = min(num_workers, len(batches))
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        tasks = [(batch, output_dir) for batch in batches]
        for result in executor.map(_plot_variable_batch, tasks):
            if result:
                continue

def sora(peb_file, perf_file, alloc_file, output_file):
    start_time,end_time,flow_variables = File_Paser.load_variables(alloc_file)
    print(f"[INFO] Loaded {len(flow_variables)} variable records")
    

    flow_time_addr = File_Paser.load_pebs_file(flow_variables, peb_file)
    print(f"[INFO] Loaded pebs from file: {peb_file}")
    
    flow_performance_pridiction = File_Paser.load_stall_perf_file(perf_file)
    aol,slowdown = flow_performance_pridiction
    

    perf_interval  = int(int(end_time)-int(start_time))/len(aol)
    new_ts = [t for t in range(int(start_time), int(end_time), int(perf_interval))]
    new_ts = new_ts[:-1]

    print(f"[INFO] Loaded perf stall from file: {perf_file}")
    
    get_access_ratio_flow(new_ts,flow_variables, flow_time_addr)
    print(f"[INFO] Get access ratio by Fo and Fm")

    flow_variables = get_page_sore_flow(new_ts,flow_variables, flow_performance_pridiction)
    
    all_page_scores = [page.score for variable in flow_variables for page in variable.pages if variable.size > PAGE_SIZE]
    avg_score_threshold = sum(all_page_scores) / len(all_page_scores) if all_page_scores else 0.0
    avg_score_threshold  /= 100.0  # 调整阈值，避免过高
    print(f"[INFO] Average page score threshold: {avg_score_threshold}")

    if not os.path.exists(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    i = 0    
    bitmap_dir = os.path.dirname(output_file)
    bitmap_file = os.path.join(bitmap_dir, f"_bitmap.txt")
    with open(output_file, 'w') as f:
        with open(bitmap_file, 'w') as bf:
            for variable in flow_variables:
                if variable.size <= PAGE_SIZE:
                    continue
            
                bitmap = [1 if page.score > avg_score_threshold else 0 for page in variable.pages]
                if all(b == 0 for b in bitmap) or all(b == 1 for b in bitmap):
                    print(f"[WARN] Variable_{i}: {variable.var_id} bitmap is all {bitmap[0]}")
                else:
                    bf.write(f"Variable_{i}: {variable.var_id}, Bitmap: {bitmap}\n".join(map(str, bitmap)) + '\n')
                for page in variable.pages:
                    f.write(f"Variable_{i}: {variable.var_id}, Page Index: {page.idx}, "
                        f"Access Ratio: {page.access_ratio}, Score: {page.score}\n")
                i += 1

    print(f"[INFO] Results written to {output_file}")

    plots_dir = os.path.join(os.path.dirname(output_file.split('.')[0]), 'plots')
    plot_variable_page_scores(flow_variables, plots_dir)
    print(f"[INFO] Plots written to {plots_dir}")

    

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python3 analyze_by_page.py <peb.txt> <perf.data> <alloc_info.txt> <output.txt>")
        sys.exit(1)
    peb_file = sys.argv[1]
    perf_file = sys.argv[2]
    alloc_file = sys.argv[3]
    output_file = sys.argv[4]
    sora(peb_file, perf_file, alloc_file, output_file) 