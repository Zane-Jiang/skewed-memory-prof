#!/bin/bash

OUT_RAW_DIR="perf_raw_data"         


init_output_dir() {
  local path="$1"
  mkdir -p "$OUT_RAW_DIR/$path"
}

perf_full_collect() {
  local program="$1"
  shift
if [ "$(id -u)" -ne 0 ]; then
    echo "Warning: You are not root. Some events may be missing."
fi
  echo "[INFO] Starting comprehensive profiling..."
  dir=$(basename "$program")
  init_output_dir "$dir"

perf record \
  -e cpu/mem-loads/P \
  -e cpu/mem-stores/ \
  -c 1 \
  -o "$OUT_RAW_DIR/"$dir"/raw.data" \
  -- "$program" "$@"

  perf script -i "$OUT_RAW_DIR/"$dir"/raw.data" > "$OUT_RAW_DIR/"$dir"/raw.txt"
  echo "[INFO] perf data collected."
}