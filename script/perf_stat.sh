#!/bin/bash

OUT_RAW_DIR="perf_raw_data"         

init_output_dir() {
  local path="$1"
  mkdir -p "$OUT_RAW_DIR/$path"
}

perf_collect_mem_load_store() {
  local program="$1"
  shift
if [ "$(id -u)" -ne 0 ]; then
    echo "Warning: You are not root. Some events may be missing."
fi
  echo "[INFO] Starting [mem_load_store] profiling..."
  dir=$(basename "$program")
  init_output_dir "$dir"
  # -c 100

  echo 3000000 > /proc/sys/kernel/perf_event_max_sample_rate
  perf mem record  \
  -o "$OUT_RAW_DIR/"$dir"/raw.data" \
  -- "$program" "$@"


  perf script -i "$OUT_RAW_DIR/"$dir"/raw.data" > "$OUT_RAW_DIR/"$dir"/raw.txt"
  echo "[INFO] perf data collected."
  mkdir -p result
  echo "running analyze_mem_load_store.py"
  python3 /home/jz/PCXL/benchmark/skewed-memory-prof/script/analyze_mem_load_store.py "$OUT_RAW_DIR/"$dir"/raw.txt" "$HEAP_PROF_PATH" "result/"$dir".txt"
}

perf_collect_l3_store_miss() {
  local program="$1"
  shift
  if [ "$(id -u)" -ne 0 ]; then
    echo "Warning: You are not root. Some events may be missing."
  fi
  echo "[INFO] Starting [l3_store_miss] profiling..."
  dir=$(basename "$program")
  init_output_dir "$dir"
  perf record -e mem_load_retired.l3_miss_ps:P  -d -o "$OUT_RAW_DIR/"$dir"/raw.data" -- "$program" "$@"
  perf script -i "$OUT_RAW_DIR/"$dir"/raw.data"  -F comm,time,event,sym,addr >  "$OUT_RAW_DIR/"$dir"/raw.txt"
  echo "[INFO] perf data collected."
  mkdir -p result
  echo "running analyze_l3_store_miss.py"
  python3 /home/jz/PCXL/benchmark/skewed-memory-prof/script/analyze_l3_store_miss.py "$OUT_RAW_DIR/"$dir"/raw.txt" "$HEAP_PROF_PATH" "result/"$dir".txt"
}


