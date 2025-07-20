#!/bin/bash
source script/perf_stat.sh
compile(){
    export PCXL_ROOT=/home/jz/PCXL
    export CC=${PCXL_ROOT}/llvm-project/build/bin/clang
    export CXX=${PCXL_ROOT}/llvm-project/build/bin/clang++
    export LD_LIBRARY_PATH=${PCXL_ROOT}/lib/:$LD_LIBRARY_PATH
    export CLANG_MODE=INSTRUMENT

    pushd benchmark
    make clean
    make -j$(nproc)
    popd
}


run_array(){
    export HEAP_PROF_PATH=heap_alloc_info/array.prof
    perf_full_collect ./benchmark/array
    sudo python3 ./script/analyze_mem_access.py perf_raw_data/array/raw.txt  $HEAP_PROF_PATH result/array.txt
    echo "[info] run array done ,save to result/array.txt"
}



run_pointer_chase(){
    export HEAP_PROF_PATH=heap_alloc_info/pointer_chase.prof
    perf_full_collect ./benchmark/pointer_chase
    sudo python3 ./script/analyze_mem_access.py perf_raw_data/pointer_chase/raw.txt  $HEAP_PROF_PATH result/pointer_chase.txt
    echo "[info] run array done ,save to result/pointer_chase.txt"
}

mkdir -p heap_alloc_info
mkdir -p result
compile
run_array
run_pointer_chase



