#!/bin/bash
PERF="/usr/bin/perf"
OUT_RAW_DIR="rst/perf_data"         

init_output_dir() {
  local path="$1"
  mkdir -p "$OUT_RAW_DIR/$path"
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

  
  "$program" "$@" 2>&1 > ${OUT_RAW_DIR}/"$dir"/log &
  cpid=$!
  sleep 1
  gpid=$cpid
  echo "[INFO] gpid[$gpid],start perf record."


  $PERF record -d  -e mem_load_retired.l3_miss:pp  -e pebs:pebs -c 3001 -p $gpid -o ${OUT_RAW_DIR}/"$dir"/pebs.data &
  pebs_pid=$!

  perf_events="cycles,CYCLE_ACTIVITY.STALLS_L3_MISS"
  perf_events="${perf_events}"",OFFCORE_REQUESTS.DEMAND_DATA_RD,OFFCORE_REQUESTS_OUTSTANDING.CYCLES_WITH_DEMAND_DATA_RD"
  echo "[INFO] start perf stat."
  $PERF stat -e ${perf_events} -I 1000 -p $gpid -o ${OUT_RAW_DIR}/"$dir"/perf.data &

  echo "[INFO] wait process"
  wait $cpid
  wait $pebs_pid

  cd ${OUT_RAW_DIR}/"$dir"/;
  pwd
  echo "[info] perf script pebs"
  $PERF script -i pebs.data > pebs.txt
  cd -;

  echo "[INFO] perf data collected."

  mkdir -p rst/analyze
  echo "running analyze_l3_store_miss.py"
  python3 /home/jz/PCXL/benchmark/skewed-memory-prof/script/analyze_l3_store_miss.py "$OUT_RAW_DIR/"$dir"/raw.txt" "$HEAP_PROF_PATH" "result/"$dir".txt"
}


