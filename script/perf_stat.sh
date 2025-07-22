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

  perf mem record -a \
  -o "$OUT_RAW_DIR/"$dir"/raw.data" \
  -- "$program" "$@"

  perf script -i "$OUT_RAW_DIR/"$dir"/raw.data" > "$OUT_RAW_DIR/"$dir"/raw.txt"
  echo "[INFO] perf data collected."
}

perf_collect_cycle_activity_stall_l3_miss() {
  local program="$1"
  shift
  if [ "$(id -u)" -ne 0 ]; then
    echo "Warning: You are not root. Some events may be missing."
  fi
  echo "[INFO] Starting [cycle_activity_stall_l3_miss] profiling..."
  dir=$(basename "$program")
  init_output_dir "$dir"
  perf record -e CYCLE_ACTIVITY.STALLS_L3_MISS -a -o "$OUT_RAW_DIR/"$dir"/raw.data" -- "$program" "$@"
  perf script -i "$OUT_RAW_DIR/"$dir"/raw.data" > "$OUT_RAW_DIR/"$dir"/raw.txt"
  echo "[INFO] perf data collected."
}
