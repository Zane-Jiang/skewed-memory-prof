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


PAGE_SIZE = 4*1024
class Page:
    def __init__(self,idx,start_addr):
        self.access_count = 0
        self.idx = idx
        self.start_addr = start_addr
        self.access_ratio = 0.0
        self.score = 0.0
        
    def contains(self,addr):
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

events = ["cycles", "CYCLE_ACTIVITY.STALLS_L3_MISS", \
  "OFFCORE_REQUESTS.DEMAND_DATA_RD", \
  "OFFCORE_REQUESTS_OUTSTANDING.CYCLES_WITH_DEMAND_DATA_RD"]
class File_Paser:
    @staticmethod
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
        # Process the file content here
            csv_header = csv.reader(csv_file, delimiter=' ')
            line_count = 0
            for row in csv_header:
                line_count += 1
                processed_row = [item for item in row if item]
                for event in events:
                    if event in processed_row:
                        value = float(processed_row[1].replace(',', ''))
                        perf_data[event].append(value)
                
        demand_data_rd_idx = events.index("OFFCORE_REQUESTS.DEMAND_DATA_RD")
        cyc_demand_data_rd_idx = events.index("OFFCORE_REQUESTS_OUTSTANDING.CYCLES_WITH_DEMAND_DATA_RD")
        l3_stall_idx = events.index("CYCLE_ACTIVITY.STALLS_L3_MISS")
        cyc_idx = events.index("cycles")

        demand_data_rd = np.asarray(perf_data[events[demand_data_rd_idx]])
        cyc_demand_data_rd = np.asarray(perf_data[events[cyc_demand_data_rd_idx]])
        l3_stall = np.asarray(perf_data[events[l3_stall_idx]])
        cyc = np.asarray(perf_data[events[cyc_idx]])        

        a = 24.67
        b = 0.87
        l3_stall_per_cyc = l3_stall/cyc 
        aol = cyc_demand_data_rd/demand_data_rd  #aol
        slowdown = [l3_stall_per_cyc[i]/(a/aol[i]+b) for i in range(len(aol))] ## 预测减速

        flow_performance_pridict = l3_stall_per_cyc,aol,slowdown
        return flow_performance_pridict

def getFactor(aol):
    factor = 1
    min_ratio = 0.03
    max_ratio = 0.7
    if aol < 80 and aol > 60:
        factor = 2
        min_ratio = 0.4
        max_ratio = 0.6
    elif aol <= 60 and aol > 45:
        factor = 4
    elif aol > 40:
        factor = 8
    else:
        factor = 12
    ret = factor,min_ratio,max_ratio
    return ret

def score(access_Ratio,predicted_perf,aol):
    factor,min_ratio,max_ratio = getFactor(aol)
    if access_Ratio >= max_ratio:
        score = access_Ratio * predicted_perf / factor
    elif access_Ratio >= min_ratio:
        score = access_Ratio * predicted_perf 
    else:
        score = access_Ratio * predicted_perf * factor
    return score


#todo 用时间片访问比例替换全局访问比例
def get_access_ratio_flow(flow_variables,flow_time_addr):
    total_access_count = 0
    for time, address in flow_time_addr:
        for variable in flow_variables:
            if variable.in_lifetime(time) and variable.contains(address):
                for page in variable.pages:
                    if page.contains(address):
                        page.access_count += 1
                        total_access_count += 1
    for variable in flow_variables:
        for page in variable.pages:
            page.access_ratio = page.access_count / total_access_count
    return  








def get_page_sore_flow(flow_variables, flow_performance_pridiction):
    l3_stall_per_cyc, aol, slowdown = flow_performance_pridiction
    
    # 遍历每个变量的页面
    for variable in flow_variables:
        total_page_score = 0
        valid_time_slices = 0
        
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
                    
                    # 计算页面在当前时间片的得分
                    if page.access_ratio >= max_ratio:
                        time_slice_score = page.access_ratio * slowdown[i] / factor
                    elif page.access_ratio >= min_ratio:
                        time_slice_score = page.access_ratio * slowdown[i]
                    else:
                        time_slice_score = page.access_ratio * slowdown[i] * factor
                    
                    page_time_scores.append(time_slice_score)
            
            # 计算页面的平均得分
            if page_time_scores:
                page.score = sum(page_time_scores) / len(page_time_scores)
                total_page_score += page.score
                valid_time_slices += 1
        
        # 如果变量在多个时间片内有效，计算变量的平均得分
        if valid_time_slices > 0:
            variable.score = total_page_score / valid_time_slices
    
    return flow_variables

def sora(peb_file, perf_file, alloc_file, output_file):
    flow_variables = File_Paser.load_variables(alloc_file)
    print(f"[INFO] Loaded {len(flow_variables)} variable records")
    
    flow_time_addr = File_Paser.load_pebs_file(flow_variables, peb_file)
    print(f"[INFO] Loaded pebs from file: {peb_file}")
    
    flow_performance_pridiction = File_Paser.load_stall_perf_file(perf_file)
    print(f"[INFO] Loaded perf stall from file: {perf_file}")
    
    get_access_ratio_flow(flow_variables, flow_time_addr)
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
        print("Usage: python3 analyze_l3_miss_stall.py <peb.txt> <perf.data> <alloc_info.txt> <output.txt>")
        sys.exit(1)
    peb_file = sys.argv[1]
    perf_file = sys.argv[2]
    alloc_file = sys.argv[3]
    output_file = sys.argv[4]
    sora(peb_file, perf_file, alloc_file, output_file) 