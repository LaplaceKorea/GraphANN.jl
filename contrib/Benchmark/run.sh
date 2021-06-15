#!/bin/zsh

# At Most One Thread Per Core
#
for i in {1..22}
do
    cpu_upper=$((21 + $i))
    echo $cpu_upper
    export JULIA_NUM_THREADS=$i
    numactl --physcpubind=22-${cpu_upper} --membind=1 ~/tools/julia-1.6.0/bin/julia --project -e "using Benchmark; Benchmark.go()"
done

# At Potentially more than one threads per core
# for i in {1..22}
# do
#     cpu_upper=$((65 + $i))
#     echo $cpu_upper
#     export JULIA_NUM_THREADS=$((22 + i))
#     numactl --physcpubind=22-43,66-${cpu_upper} --membind=1 ~/tools/julia-1.6.0/bin/julia --project -e "using Benchmark; Benchmark.go()"
# done
