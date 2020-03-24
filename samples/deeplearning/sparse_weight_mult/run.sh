#!/bin/bash

N=${1:-160}
C=${2:-1024}
K=${3:-1024}
spar=${4:-0.9}
rep=${5:-30}
echo "N        = " $N
echo "C        = " $C
echo "K        = " $K
echo "Sparsity = " $spar
echo "Repeats  = " $rep
fwd_max_perf=0
upd_max_perf=0
#for NB in 16 32 80 160 320
for NB in 160 320
do
  #for nb in 16 32 80 160 320
  for nb in 16 32
  do
    if [[ "$nb" -gt "$NB" ]]; then
      continue
    fi
    if [[ "$nb" == "32" && "$NB" == "80"  ]]; then
      continue
    fi
    #for KB in 8 16 32 64 128
    for KB in 128
    do
      #for CB in 8 16 32 64 128
      for CB in 64 128
      do
        echo "NB =" $NB ", nb =" $nb ", CB =" $CB ", KB =" $KB
        KMP_AFFINITY=compact,granularity=fine,1,28 OMP_NUM_THREADS=28 srun ./parallel_sparse_weight_B_mult $N $C $K $NB $CB $KB $nb $spar $rep 2>&1 1>tmp
        if [ $? -eq 0 ]
        then
          fwd_perf=$(grep "GFLOPS " tmp | awk -F " " '{print $1}')
          fwd_perf=${fwd_perf%.*}
          echo "  FWD_PERF =" $fwd_perf
          fwd_max_perf=$(( fwd_perf > fwd_max_perf ? fwd_perf : fwd_max_perf ))
        else
          echo "  FWD Fail"
        fi
        rm -f tmp
        KMP_AFFINITY=compact,granularity=fine,1,28 OMP_NUM_THREADS=28 srun ./parallel_sparse_weight_C_redmult $N $C $K $NB $CB $KB $nb $spar $rep 2>&1 1>tmp
        if [ $? -eq 0 ]
        then
          upd_perf=$(grep "GFLOPS " tmp | awk -F " " '{print $1}')
          upd_perf=${upd_perf%.*}
          echo "  UPD_PERF =" $upd_perf
          upd_max_perf=$(( upd_perf > upd_max_perf ? upd_perf : upd_max_perf ))
        else
          echo "  UPD Fail"
        fi
        rm -f tmp
      done
    done
  done
done
echo "FWD_MAX_PERF ="  $fwd_max_perf "GFLOPS"
echo "UPD_MAX_PERF ="  $upd_max_perf "GFLOPS"