#!/bin/bash

if [ $# -ne 3 ]
then
  echo "Usage: $(basename $0) matrices.txt iters numa (1-mcdram/0-DDR)"
  fn="deepbench_matrices.txt"
  ITERS=100
  NUMA=1
else
  fn=$1
  ITERS=$2
  NUMA=$3
fi

NUMACTL="${TOOL_COMMAND}"
CPUFLAGS=$(if [ -e /proc/cpuinfo ]; then grep -m1 flags /proc/cpuinfo | cut -d: -f2-; fi)
if [ "" != "$(echo "${CPUFLAGS}" | grep -o avx512er)" ]; then
  if [ "0" != "$((NUMA < $(numactl -H | grep "node  " | tr -s " " | cut -d" " -f2- | wc -w)))" ]; then
    NUMACTL="numactl --membind=${NUMA} ${TOOL_COMMAND}"
  fi
fi

if [[ -z "${OMP_NUM_THREADS}" ]]; then
  echo "using defaults for OMP settings!"
  export KMP_HW_SUBSET=1T
  export KMP_AFFINITY=compact,granularity=fine
  export OMP_NUM_THREADS=64
else
  echo "using environment OMP settings!"
fi

#./block_gemm iters M N K order BM BN BK B_M B_N B_K K_unroll
# current block_gemm only supports non-tranposne GEMM, TODO to transpose support
_bin="$NUMACTL ./block_gemm"
_it=$ITERS
function run_bsgemm {
_M=$1
_N=$2
_K=$3
_AT=$4
_BT=$5
_order=0
_mb1=1
_nb1=1
_kb1=1
_kb2=1
if [[ $# -gt 5 ]]
then
  _mb=$6
  _nb=$7
  _kb=$8
else
  _mb=32
  _nb=32
  _kb=32
fi

if [[ "$_mb" -gt "_M" ]]
then
  _mb=$_M
fi
if [[ "$_nb" -gt "_N" ]]
then
  _nb=$_N
fi
if [[ "$_kb" -gt "_K" ]]
then
  _kb=$_K
fi

if [[ "$((_M % _mb))" -gt 0 ]]
then
  #_mb=$_M
  _M=$((_mb*(_M/_mb+1)))
fi
if [[ "$((_N % _nb))" -gt 0 ]]
then
  #_nb=$_N
  _N=$((_nb*(_N/_nb+1)))
fi
if [[ "$((_K % _kb))" -gt 0 ]]
then
  #_kb=$_K
  _K=$((_kb*(_K/_kb+1)))
fi

if [[ "$_AT" == "T" ]]
then
  echo "!!! $_M $_N $_K $_AT $_BT - Not supported !!!, doing $_N $_M $_K N N instead"
  t_M=$_M
  t_mb=$_mb
  _M=$_N
  _N=$t_M
  _mb=$_nb
  _nb=$t_mb
fi
if [[ "$_BT" == "T" ]]
then
  echo "!!! $_M $_N $_K $_AT $_BT - Not supported !!!, doing $_M $_K $_N N N instead"
  t_K=$_K
  t_kb=$_kb
  _K=$_N
  _N=$t_K
  _kb=$_nb
  _nb=$t_kb
fi

echo "$_bin $_it $_M $_N $_K $_order $_mb $_nb $_kb $_mb1 $_nb1 $_kb1 $_kb2"
$_bin $_it $_M $_N $_K $_order $_mb $_nb $_kb $_mb1 $_nb1 $_kb1 $_kb2 
echo "--------------------------------------------------------------------------------------"
}

nc=$(wc -l $fn)
idx=1

cat $fn | while read line
do
  if [ ! -z "$line" ]; then
    echo -n "($idx/$nc)  "
    run_bsgemm $line 
  fi
  idx=$((idx+1))
done
