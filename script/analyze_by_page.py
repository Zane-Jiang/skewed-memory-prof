#!/usr/bin/env python3

import re
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import glob
from collections import OrderedDict
import csv

events = ["cycles", "CYCLE_ACTIVITY.STALLS_L3_MISS", 
  "OFFCORE_REQUESTS.DEMAND_DATA_RD", 
  "OFFCORE_REQUESTS_OUTSTANDING.CYCLES_WITH_DEMAND_DATA_RD"]
demand_data_rd_idx = events.index("OFFCORE_REQUESTS.DEMAND_DATA_RD")
cyc_demand_data_rd_idx = events.index("OFFCORE_REQUESTS_OUTSTANDING.CYCLES_WITH_DEMAND_DATA_RD")
l3_stall_idx = events.index("CYCLE_ACTIVITY.STALLS_L3_MISS")
cyc_idx = events.index("cycles")

PAGE_SIZE = 4*1024
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
    def __init__(self, var_id, ptr, location, size, alloc_ts, free_ts):
        self.var_id = var_id
        self.start_addr = int(ptr, 16)
        self.end_addr = self.start_addr + int(size)
        self.size = int(size)
        self.alloc_ts = float(alloc_ts)
        self.free_ts = float(free_ts)
        self.location = location

        end_pg   = (self.size) // PAGE_SIZE
        self.pages = []
        for p in range(0, end_pg + 1):
            self.pages.append(Page(idx=p, start_addr=self.start_addr + p * PAGE_SIZE))


    def in_lifetime(self, ts):
        return self.alloc_ts <= ts <= self.free_ts

    def contains(self, addr):
        return self.start_addr <= addr < self.end_addr

class File_Paser:
    @staticmethod
    def load_variables(alloc_file):
        start_time = 0
        end_time = 0
        variables = []
        with open(alloc_file, 'r') as f:
            #todo 
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

        flow_performance_pridiction = l3_stall_per_cyc,aol,slowdown
        return flow_performance_pridiction


def get_access_ratio_flow(new_ts, flow_variables, flow_time_addr):
    # 为每个变量和页面初始化时间片访问计数
    for variable in flow_variables:
        for page in variable.pages:
            page.time_slice_access_counts = [0] * len(new_ts)
            page.time_slice_access_ratios = [0.0] * len(new_ts)

    # 遍历每个时间地址对
    for time, address in flow_time_addr:
        # 找到对应的时间片索引
        time_slice_idx = next((i for i, ts_end in enumerate(new_ts) if time < ts_end), len(new_ts) - 1)
        
        # 检查地址是否属于某个变量和页面
        for variable in flow_variables:
            if variable.in_lifetime(time) and variable.contains(address):
                for page in variable.pages:
                    if page.contains(address):
                        page.time_slice_access_counts[time_slice_idx] += 1

    # 计算每个时间片的总访问次数
    time_slice_total_access_counts = [0] * len(new_ts)
    for variable in flow_variables:
        for page in variable.pages:
            for i in range(len(new_ts)):
                time_slice_total_access_counts[i] += page.time_slice_access_counts[i]

    # 计算每个页面在每个时间片的访问比例
    for variable in flow_variables:
        for page in variable.pages:
            for i in range(len(new_ts)):
                # 避免除零错误
                if time_slice_total_access_counts[i] > 0:
                    page.time_slice_access_ratios[i] = page.time_slice_access_counts[i] / time_slice_total_access_counts[i]
                else:
                    page.time_slice_access_ratios[i] = 0.0

            page.access_count = sum(page.time_slice_access_counts)
            page.access_ratio = page.access_count / sum(time_slice_total_access_counts) if sum(time_slice_total_access_counts) > 0 else 0.0

    return








def get_page_sore_flow(flow_variables, flow_performance_pridiction):
    l3_stall_per_cyc, aol, slowdown = flow_performance_pridiction
    
    # 遍历每个变量的页面
    for variable in flow_variables:
        
        # 遍历每个页面
        for page in variable.pages:
            page_time_scores = []
            
            # 遍历每个时间片
            for i in range(len(aol)):
                # 检查页面是否在当前时间片的生命周期内
                if variable.in_lifetime(i):
                    # 根据AOL确定规模因子
                    factor = 1
                    min_ratio = 0.03
                    max_ratio = 0.7
                    
                    if aol[i] < 80 and aol[i] > 60:
                        factor = 2
                        min_ratio = 0.4
                        max_ratio = 0.6
                    elif aol[i] <= 60 and aol[i] > 45:
                        factor = 4
                    elif aol[i] > 40:
                        factor = 8
                    else:
                        factor = 12
                    
                    # 使用时间片访问比例计算得分
                    access_ratio = page.time_slice_access_ratios[i] if i < len(page.time_slice_access_ratios) else 0.0
                    
                    # 计算页面在当前时间片的得分
                    if access_ratio >= max_ratio:
                        time_slice_score = access_ratio * slowdown[i] / factor
                    elif access_ratio >= min_ratio:
                        time_slice_score = access_ratio * slowdown[i]
                    else:
                        time_slice_score = access_ratio * slowdown[i] * factor
                    
                    page_time_scores.append(time_slice_score)
            
            # 计算页面的总得分
            if page_time_scores:
                page.score = sum(page_time_scores) 
    return flow_variables

def sora(peb_file, perf_file, alloc_file, output_file):
    start_time,end_time,flow_variables = File_Paser.load_variables(alloc_file)
    print(f"[INFO] Loaded {len(flow_variables)} variable records")
    
    flow_time_addr = File_Paser.load_pebs_file(flow_variables, peb_file)
    print(f"[INFO] Loaded pebs from file: {peb_file}")
    
    flow_performance_pridiction = File_Paser.load_stall_perf_file(perf_file)
    l3_stall_per_cyc,aol,slowdown = flow_performance_pridiction
    

    perf_interval  = (end_time-start_time)/len(l3_stall_per_cyc)
    new_ts = [t for t in range(int(start_time), int(end_time), perf_interval)]
    new_ts = new_ts[:-1]

    print(f"[INFO] Loaded perf stall from file: {perf_file}")
    
    get_access_ratio_flow(new_ts,flow_variables, flow_time_addr)
    print(f"[INFO] Get access ratio by Fo and Fm")

    flow_variables = get_page_sore_flow(flow_variables, flow_performance_pridiction)
    
    # 输出结果到文件
    with open(output_file, 'w') as f:
        for variable in flow_variables:
            for page in variable.pages:
                f.write(f"Variable: {variable.var_id}, Page Index: {page.idx}, "
                        f"Access Ratio: {page.access_ratio}, Score: {page.score}\n")
    
    print(f"[INFO] Results written to {output_file}")

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python3 analyze_by_page.py <peb.txt> <perf.data> <alloc_info.txt> <output.txt>")
        sys.exit(1)
    peb_file = sys.argv[1]
    perf_file = sys.argv[2]
    alloc_file = sys.argv[3]
    output_file = sys.argv[4]
    sora(peb_file, perf_file, alloc_file, output_file) 